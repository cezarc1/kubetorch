from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from qwen3_asr_orin.datasets import AudioExample, load_manifest, write_manifest


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
    write_manifest(bundled_examples, bundled_manifest_path)
    prompts_path.write_text(
        "".join(f"{example.id}\t{example.transcript}\n" for example in bundled_examples),
        encoding="utf-8",
    )
    pipeline_path.write_text(
        json.dumps(
            _trt_edgellm_pipeline(
                model_path=model_path,
                qwen_asr_root=qwen_asr_root,
                quant_format=quant_format,
                output_dir=output_dir,
                audio_dir=audio_dir,
                prompts_path=prompts_path,
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
    trt_edgellm_root: str,
) -> dict:
    quantized_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}"
    onnx_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}-ONNX"
    engine_dir = output_dir / f"Qwen3-ASR-1.7B-{quant_format}-Engines"
    result_dir = output_dir / f"results-{quant_format}"
    return {
        "quantization": {
            "format": quant_format,
            "model_path": model_path,
            "qwen_asr_root": qwen_asr_root,
            "quantized_dir": str(quantized_dir),
            "onnx_dir": str(onnx_dir),
            "engine_dir": str(engine_dir),
            "trt_edgellm_root": trt_edgellm_root,
        },
        "notes": [
            "Quantize and export on the 4090/x86 host.",
            "Build TensorRT engines on an Orin-class Jetson with enough memory; Nano 8 GB may not build engines reliably.",
            "Run benchmark/inference on jetson-orin-nano-01 only for reportable Orin numbers.",
        ],
        "stages": [
            {
                "name": "quantize",
                "platform": "x86_64-4090",
                "command": (
                    f"bash scripts/01_quantize.sh --model_path {model_path} "
                    f"--qwen_asr_root {qwen_asr_root} --format {quant_format} --output_dir {quantized_dir}"
                ),
            },
            {
                "name": "export-onnx",
                "platform": "x86_64-4090",
                "command": f"bash scripts/02_export_onnx.sh {quantized_dir} {onnx_dir}",
            },
            {
                "name": "build-engine",
                "platform": "jetson-orin-builder",
                "command": f"bash scripts/03_build_engine.sh {onnx_dir} {engine_dir}",
            },
            {
                "name": "benchmark-nano",
                "platform": "jetson-orin-nano-01",
                "command": (
                    f"bash scripts/04_benchmark.sh --audio_dir {audio_dir} --prompts {prompts_path} "
                    f"--engine_dir {engine_dir} --work_dir {result_dir} --trt_edgellm_dir {trt_edgellm_root}"
                ),
            },
        ],
    }
