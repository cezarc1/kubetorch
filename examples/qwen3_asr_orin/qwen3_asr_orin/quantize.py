from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from qwen3_asr_orin.datasets import AudioExample, load_manifest, write_manifest

TENSORRT_EDGE_LLM_VERSION = "v0.8.0"
TENSORRT_EDGE_LLM_COMMIT = "f9cc74623d95d7acf1addab6026b9d410ba81f52"


@dataclass(frozen=True)
class QuantizationSpec:
    model_id: str
    output_dir: str
    manifest_path: str
    algorithm: str = "smoothquant"
    precision: str = "int8"
    quantized_modules: str = "llm_decoder"
    preserved_modules: tuple[str, ...] = ("*audio*", "*audio_tower*", "*lm_head*")


@dataclass(frozen=True)
class TrtEdgeLlmBundle:
    output_dir: str
    audio_dir: str
    manifest_path: str
    prompts_path: str
    pipeline_path: str
    sample_count: int


def int8_smoothquant_config(preserved_patterns: tuple[str, ...] = ("*audio*", "*audio_tower*", "*lm_head*")) -> dict:
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "enable": True, "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "enable": True, "cfg": {"num_bits": 8, "axis": None}},
    ]
    quant_cfg.extend({"quantizer_name": pattern, "enable": False} for pattern in preserved_patterns)
    return {"algorithm": "smoothquant", "quant_cfg": quant_cfg}


def write_quantization_spec(spec: QuantizationSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "quantization-spec.json"
    payload = asdict(spec)
    payload["modelopt_config"] = int8_smoothquant_config(spec.preserved_modules)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec_path


def run_modelopt_quantization(spec: QuantizationSpec) -> Path:
    """Record a ModelOpt INT8 spec and fail clearly before unsafe fake quantization.

    Qwen3-ASR's remote-code forward path needs audio preprocessing and export
    validation. This first implementation writes a reproducible ModelOpt config
    for the Kubetorch run artifact, then stops before producing an artifact that
    would look deployable without TensorRT validation.
    """
    output_dir = Path(spec.output_dir)
    spec_path = write_quantization_spec(spec, output_dir)
    raise RuntimeError(
        "ModelOpt INT8 spec written to "
        f"{spec_path}. Full Qwen3-ASR calibration/export still requires a validated audio forward_loop."
    )


def export_trt_edgellm_bundle(
    manifest_path: Path,
    output_dir: Path,
    model_path: str,
    qwen_asr_root: str,
    quant_format: str = "int8",
    materialize_audio: str = "symlink",
    trt_edgellm_root: str = "$TRT_EDGELLM_ROOT",
) -> TrtEdgeLlmBundle:
    """Create TensorRT-Edge-LLM input files from the benchmark manifest."""
    if quant_format not in {"int8", "int4"}:
        raise ValueError("quant_format must be int8 or int4")
    if materialize_audio not in {"copy", "symlink"}:
        raise ValueError("materialize_audio must be copy or symlink")

    examples = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    bundled_examples: list[AudioExample] = []
    for example in examples:
        source = Path(example.audio_path)
        target = audio_dir / _bundle_audio_name(example, source)
        _materialize_audio(source, target, materialize_audio)
        bundled_examples.append(
            AudioExample(
                id=example.id,
                dataset=example.dataset,
                language=example.language,
                split=example.split,
                audio_path=str(target),
                transcript=example.transcript,
                duration_seconds=example.duration_seconds,
            )
        )

    bundled_manifest_path = output_dir / "manifest.jsonl"
    prompts_path = output_dir / "prompts.txt"
    pipeline_path = output_dir / "pipeline.json"
    benchmark_script_path = output_dir / "scripts" / "qwen3_asr_edgellm_benchmark.py"
    write_manifest(bundled_examples, bundled_manifest_path)
    prompts_path.write_text(
        "".join(f"{example.id}\t{example.transcript}\n" for example in bundled_examples),
        encoding="utf-8",
    )
    _write_edgellm_benchmark_script(benchmark_script_path)
    pipeline_path.write_text(
        json.dumps(
            _trt_edgellm_pipeline(
                model_path=model_path,
                qwen_asr_root=qwen_asr_root,
                quant_format=quant_format,
                output_dir=output_dir,
                audio_dir=audio_dir,
                prompts_path=prompts_path,
                benchmark_script_path=benchmark_script_path,
                trt_edgellm_root=trt_edgellm_root,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrtEdgeLlmBundle(
        output_dir=str(output_dir),
        audio_dir=str(audio_dir),
        manifest_path=str(bundled_manifest_path),
        prompts_path=str(prompts_path),
        pipeline_path=str(pipeline_path),
        sample_count=len(bundled_examples),
    )


def _bundle_audio_name(example: AudioExample, source: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", example.id).strip("._") or "audio"
    suffix = source.suffix or ".wav"
    return f"{stem}{suffix}"


def _materialize_audio(source: Path, target: Path, materialize_audio: str) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        target.unlink()
    if materialize_audio == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


def _trt_edgellm_pipeline(
    model_path: str,
    qwen_asr_root: str,
    quant_format: str,
    output_dir: Path,
    audio_dir: Path,
    prompts_path: Path,
    benchmark_script_path: Path,
    trt_edgellm_root: str,
) -> dict:
    quantized_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}"
    onnx_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}-ONNX"
    engine_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}-Engines"
    result_dir = output_dir / f"results-{quant_format}"
    return {
        "quantization": {
            "format": quant_format,
            "tensorrt_edgellm_version": TENSORRT_EDGE_LLM_VERSION,
            "tensorrt_edgellm_commit": TENSORRT_EDGE_LLM_COMMIT,
            "model_path": model_path,
            "qwen_asr_root": qwen_asr_root,
            "quantized_dir": str(quantized_dir),
            "onnx_dir": str(onnx_dir),
            "engine_dir": str(engine_dir),
            "trt_edgellm_root": trt_edgellm_root,
        },
        "notes": [
            "Quantize and export on the 4090/x86 host with TensorRT-Edge-LLM v0.8.0.",
            "Build TensorRT engines on an Orin-class Jetson with enough memory; Nano 8 GB may not build engines reliably.",
            "Run benchmark/inference on jetson-orin-nano-01 only for reportable Orin numbers.",
        ],
        "stages": [
            {
                "name": "quantize",
                "platform": "x86_64-4090",
                "command": (
                    "tensorrt-edgellm-quantize llm "
                    f"--model_dir {model_path} "
                    f"--output_dir {quantized_dir} "
                    f"--quantization {_edgellm_quantization_method(quant_format)} "
                    "--num_samples 128"
                ),
            },
            {
                "name": "export-onnx",
                "platform": "x86_64-4090",
                "command": f"tensorrt-edgellm-export {quantized_dir} {onnx_dir}",
            },
            {
                "name": "build-llm-engine",
                "platform": "jetson-orin-builder",
                "command": (
                    "./build/examples/llm/llm_build "
                    f"--onnxDir {onnx_dir}/llm "
                    f"--engineDir {engine_dir}/llm "
                    "--maxBatchSize 1 "
                    "--maxInputLen 1024 "
                    "--maxKVCacheCapacity 4096"
                ),
            },
            {
                "name": "build-audio-engine",
                "platform": "jetson-orin-builder",
                "command": (
                    "./build/examples/multimodal/audio_build "
                    f"--onnxDir {onnx_dir}/audio "
                    f"--engineDir {engine_dir}/audio "
                    "--minTimeSteps 1000 "
                    "--maxTimeSteps 3000"
                ),
            },
            {
                "name": "preprocess-audio",
                "platform": "x86_64-4090-or-jetson-orin",
                "command": (
                    "mkdir -p "
                    f"{output_dir}/preprocessed-audio && "
                    "while IFS=$'\\t' read -r sample_id _; do "
                    f"tensorrt-edgellm-preprocess-audio --input {audio_dir}/$sample_id.* "
                    f"--output {output_dir}/preprocessed-audio/$sample_id.safetensors; "
                    f"done < {prompts_path}"
                ),
            },
            {
                "name": "benchmark-nano",
                "platform": "jetson-orin-nano-01",
                "command": (
                    f"python {benchmark_script_path} --prompts {prompts_path} "
                    f"--manifest {output_dir}/manifest.jsonl "
                    f"--preprocessed-audio-dir {output_dir}/preprocessed-audio "
                    f"--engine-dir {engine_dir} --output-dir {result_dir} "
                    f"--trt-edgellm-root {trt_edgellm_root} "
                    "--llm-inference ./build/examples/llm/llm_inference"
                ),
            },
        ],
    }


def _edgellm_quantization_method(quant_format: str) -> str:
    if quant_format == "int8":
        return "int8_sq"
    if quant_format == "int4":
        return "int4_awq"
    raise ValueError("quant_format must be int8 or int4")


def _write_edgellm_benchmark_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_EDGELLM_BENCHMARK_SCRIPT, encoding="utf-8")
    path.chmod(0o755)


_EDGELLM_BENCHMARK_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _load_prompts(path: Path) -> list[tuple[str, str]]:
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample_id, transcript = line.rstrip("\n").split("\t", 1)
            prompts.append((sample_id, transcript))
    return prompts


def _load_durations(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    durations = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            durations[str(row["id"])] = float(row["duration_seconds"])
    return durations


def _input_payload(audio_path: Path) -> dict:
    return {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": 256,
        "requests": [
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]},
                ]
            }
        ],
    }


def _first_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "output", "generated_text", "transcript"):
            found = _first_text(value.get(key))
            if found:
                return found
        for child in value.values():
            found = _first_text(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _first_text(child)
            if found:
                return found
    return ""


def _words(text: str) -> list[str]:
    return [token.lower() for token in text.replace(".", " ").replace(",", " ").split()]


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _wer(reference: str, hypothesis: str) -> float:
    reference_words = _words(reference)
    if not reference_words:
        return 0.0 if not _words(hypothesis) else 1.0
    return _edit_distance(reference_words, _words(hypothesis)) / len(reference_words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT-Edge-LLM Qwen3-ASR engines")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preprocessed-audio-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trt-edgellm-root", type=Path, default=Path("."))
    parser.add_argument("--llm-inference", type=Path, default=Path("./build/examples/llm/llm_inference"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.jsonl"
    durations = _load_durations(args.manifest)
    total_audio_seconds = 0.0
    total_latency_seconds = 0.0
    total_rtf = 0.0
    total_wer = 0.0
    errors = 0
    rows = []

    for sample_id, reference in _load_prompts(args.prompts):
        audio_path = args.preprocessed_audio_dir / f"{sample_id}.safetensors"
        input_path = args.output_dir / f"{sample_id}.input.json"
        output_path = args.output_dir / f"{sample_id}.output.json"
        input_path.write_text(json.dumps(_input_payload(audio_path), indent=2) + "\n", encoding="utf-8")
        command = [
            str(args.llm_inference),
            "--engineDir",
            str(args.engine_dir / "llm"),
            "--multimodalEngineDir",
            str(args.engine_dir / "audio"),
            "--inputFile",
            str(input_path),
            "--outputFile",
            str(output_path),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=args.trt_edgellm_root, text=True, capture_output=True, check=False)
        latency = time.perf_counter() - started
        hypothesis = ""
        if output_path.exists():
            hypothesis = _first_text(json.loads(output_path.read_text(encoding="utf-8")))
        duration_seconds = durations.get(sample_id, 0.0)
        rtf = latency / duration_seconds if duration_seconds > 0 else 0.0
        wer = _wer(reference, hypothesis)
        row = {
            "sample_id": sample_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "duration_seconds": duration_seconds,
            "latency_seconds": latency,
            "rtf": rtf,
            "wer": wer,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
        rows.append(row)
        total_audio_seconds += duration_seconds
        total_latency_seconds += latency
        total_rtf += rtf
        total_wer += wer
        if completed.returncode != 0:
            errors += 1

    with samples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "sample_count": len(rows),
        "errors": errors,
        "total_audio_seconds": total_audio_seconds,
        "total_latency_seconds": total_latency_seconds,
        "mean_latency_seconds": total_latency_seconds / len(rows) if rows else 0.0,
        "mean_rtf": total_rtf / len(rows) if rows else 0.0,
        "aggregate_rtf": total_latency_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
        "mean_wer": total_wer / len(rows) if rows else 0.0,
        "samples_path": str(samples_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
