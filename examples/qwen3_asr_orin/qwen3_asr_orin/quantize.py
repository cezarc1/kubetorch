from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from qwen3_asr_orin.datasets import AudioExample, load_manifest, write_manifest

TENSORRT_EDGE_LLM_VERSION = "v0.8.0"
TENSORRT_EDGE_LLM_COMMIT = "f9cc74623d95d7acf1addab6026b9d410ba81f52"
JETSON_NODE_NAME = "jetson-orin-nano-01"
JETSON_HOST_CUDA_PATH = "/usr/local/cuda"
JETSON_HOST_USR_PATH = "/usr"
JETSON_TEMP_SWAP_GIB = 16
JETSON_TENSORRT_VERSION = "10.16.2.10"


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
    hosttrt_runner_path: str
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
    jetson_node_name: str = JETSON_NODE_NAME,
    host_cuda_path: str = JETSON_HOST_CUDA_PATH,
    host_usr_path: str = JETSON_HOST_USR_PATH,
    temp_swap_gib: int = JETSON_TEMP_SWAP_GIB,
    runtime_output_dir: str | None = None,
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
    runtime_dir = Path(runtime_output_dir) if runtime_output_dir else output_dir
    runtime_audio_dir = runtime_dir / "audio"

    bundled_examples: list[AudioExample] = []
    for example in examples:
        source = Path(example.audio_path)
        audio_name = _bundle_audio_name(example, source)
        target = audio_dir / audio_name
        _materialize_audio(source, target, materialize_audio)
        bundled_examples.append(
            AudioExample(
                id=example.id,
                dataset=example.dataset,
                language=example.language,
                split=example.split,
                audio_path=str(runtime_audio_dir / audio_name),
                transcript=example.transcript,
                duration_seconds=example.duration_seconds,
            )
        )

    bundled_manifest_path = output_dir / "manifest.jsonl"
    prompts_path = output_dir / "prompts.txt"
    pipeline_path = output_dir / "pipeline.json"
    benchmark_script_path = output_dir / "scripts" / "qwen3_asr_edgellm_benchmark.py"
    hosttrt_runner_path = output_dir / "scripts" / "run_jetson_hosttrt_pipeline.sh"
    write_manifest(bundled_examples, bundled_manifest_path)
    prompts_path.write_text(
        "".join(f"{example.id}\t{example.transcript}\n" for example in bundled_examples),
        encoding="utf-8",
    )
    _write_edgellm_benchmark_script(benchmark_script_path)
    _write_jetson_hosttrt_runner_script(hosttrt_runner_path)
    pipeline_path.write_text(
        json.dumps(
            _trt_edgellm_pipeline(
                model_path=model_path,
                qwen_asr_root=qwen_asr_root,
                quant_format=quant_format,
                output_dir=output_dir,
                runtime_dir=runtime_dir,
                audio_dir=audio_dir,
                runtime_audio_dir=runtime_audio_dir,
                prompts_path=prompts_path,
                benchmark_script_path=benchmark_script_path,
                trt_edgellm_root=trt_edgellm_root,
                hosttrt_runner_path=hosttrt_runner_path,
                jetson_node_name=jetson_node_name,
                host_cuda_path=host_cuda_path,
                host_usr_path=host_usr_path,
                temp_swap_gib=temp_swap_gib,
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
        hosttrt_runner_path=str(hosttrt_runner_path),
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
    runtime_dir: Path,
    audio_dir: Path,
    runtime_audio_dir: Path,
    prompts_path: Path,
    benchmark_script_path: Path,
    trt_edgellm_root: str,
    hosttrt_runner_path: Path,
    jetson_node_name: str,
    host_cuda_path: str,
    host_usr_path: str,
    temp_swap_gib: int,
) -> dict:
    quantized_dir = runtime_dir / f"Qwen3-ASR-1.7B-{quant_format}"
    onnx_dir = runtime_dir / f"Qwen3-ASR-1.7B-{quant_format}-ONNX"
    engine_dir = runtime_dir / f"Qwen3-ASR-1.7B-{quant_format}-Engines"
    result_dir = runtime_dir / f"results-{quant_format}"
    runtime_prompts_path = runtime_dir / "prompts.txt"
    runtime_manifest_path = runtime_dir / "manifest.jsonl"
    runtime_benchmark_script_path = runtime_dir / "scripts" / "qwen3_asr_edgellm_benchmark.py"
    runtime_hosttrt_runner_path = runtime_dir / "scripts" / "run_jetson_hosttrt_pipeline.sh"
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
            "local_output_dir": str(output_dir),
            "runtime_output_dir": str(runtime_dir),
        },
        "runtime_requirements": {
            "benchmark_node": jetson_node_name,
            "jetpack": "7.2",
            "jetson_linux": "39.2",
            "architecture": "aarch64",
            "cuda_home": host_cuda_path,
            "host_usr_path": host_usr_path,
            "tensorrt_version": JETSON_TENSORRT_VERSION,
            "temp_swap_gib": temp_swap_gib,
            "kubernetes": {
                "node_selector": {"accelerator": "nvidia-jetson-orin-nano"},
                "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                "gpu_limit": {"nvidia.com/gpu": 1},
                "runtime_class_name": None,
                "runtime_note": "Jetson K3s agent uses Docker with NVIDIA as the default runtime.",
            },
        },
        "runner": {
            "jetson_hosttrt": str(runtime_hosttrt_runner_path),
            "local_jetson_hosttrt": str(hosttrt_runner_path),
            "usage": f"ALLOW_TEMP_SWAP=1 {runtime_hosttrt_runner_path} {runtime_dir}/pipeline.json",
        },
        "notes": [
            "Quantize and export on the 4090/x86 host with TensorRT-Edge-LLM v0.8.0.",
            "Build TensorRT engines on an Orin-class Jetson with host JetPack CUDA/TensorRT libraries.",
            "The Orin Nano 8 GB engine build needs temporary swap; the generated runner creates it only when ALLOW_TEMP_SWAP=1.",
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
                    f"{runtime_dir}/preprocessed-audio && "
                    "while IFS=$'\\t' read -r sample_id _; do "
                    f"audio_file=$(find {runtime_audio_dir} -maxdepth 1 -type f -name \"$sample_id.*\" -print -quit); "
                    'test -n "$audio_file"; '
                    'tensorrt-edgellm-preprocess-audio --input "$audio_file" '
                    f"--output {runtime_dir}/preprocessed-audio/$sample_id.safetensors; "
                    f"done < {runtime_prompts_path}"
                ),
            },
            {
                "name": "benchmark-nano",
                "platform": "jetson-orin-nano-01",
                "command": (
                    f"python {runtime_benchmark_script_path} --prompts {runtime_prompts_path} "
                    f"--manifest {runtime_manifest_path} "
                    f"--preprocessed-audio-dir {runtime_dir}/preprocessed-audio "
                    f"--engine-dir {engine_dir} --output-dir {result_dir} "
                    f"--trt-edgellm-root {trt_edgellm_root} "
                    "--llm-inference ./build/examples/llm/llm_inference "
                    "--strip-hypothesis-prefix-regex '^language\\s+\\S+\\s+'"
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


def _write_jetson_hosttrt_runner_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_JETSON_HOSTTRT_RUNNER_SCRIPT, encoding="utf-8")
    path.chmod(0o755)


_JETSON_HOSTTRT_RUNNER_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail

PIPELINE_JSON="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pipeline.json}"
LOG_DIR="${LOG_DIR:-$(dirname "${PIPELINE_JSON}")/logs/hosttrt-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${LOG_DIR}"

json_get() {
  python3 - "$PIPELINE_JSON" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

stage_command() {
  python3 - "$PIPELINE_JSON" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for stage in payload["stages"]:
    if stage["name"] == sys.argv[2]:
        print(stage["command"])
        break
else:
    raise SystemExit(f"missing stage: {sys.argv[2]}")
PY
}

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This runner must execute on an aarch64 Jetson target." >&2
  exit 2
fi

CUDA_HOME="${CUDA_HOME:-$(json_get runtime_requirements.cuda_home)}"
HOST_USR_PATH="${HOST_USR_PATH:-$(json_get runtime_requirements.host_usr_path)}"
TEMP_SWAP_GIB="$(json_get runtime_requirements.temp_swap_gib)"
TRT_EDGELLM_ROOT="${TRT_EDGELLM_ROOT:-$(pwd)}"
ENGINE_DIR="$(json_get quantization.engine_dir)"
export CUDA_HOME
export TRT_EDGELLM_ROOT
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-${TRT_EDGELLM_ROOT}/build/libNvInfer_edgellm_plugin.so}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${TRT_EDGELLM_ROOT}/build:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/sbsa-linux/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:${HOST_USR_PATH}/lib/aarch64-linux-gnu/nvidia:${HOST_USR_PATH}/lib/aarch64-linux-gnu/tegra:${HOST_USR_PATH}/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
PYTORCH_LD_LIBRARY_PATH="${TRT_EDGELLM_ROOT}/build:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/sbsa-linux/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing nvcc at ${CUDA_HOME}/bin/nvcc; mount or point CUDA_HOME at JetPack CUDA." >&2
  exit 2
fi

if ! ldconfig -p 2>/dev/null | grep -q 'libnvinfer.so.10'; then
  echo "libnvinfer.so.10 was not visible through ldconfig; continuing with LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >&2
fi

if [[ ! -f "${EDGELLM_PLUGIN_PATH}" ]]; then
  echo "Missing Edge-LLM plugin library at ${EDGELLM_PLUGIN_PATH}; build NvInfer_edgellm_plugin before running this pipeline." >&2
  exit 2
fi

SWAPFILE="${SWAPFILE:-/var/tmp/qwen3-asr-edgellm.swap}"
run_privileged() {
  if [[ "${EUID}" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root or sudo for: $*" >&2
    return 1
  fi
}

cleanup_swap() {
  if [[ "${CREATED_SWAP:-0}" == "1" ]]; then
    run_privileged swapoff "${SWAPFILE}" || true
    run_privileged rm -f "${SWAPFILE}" || true
  fi
}
trap cleanup_swap EXIT

current_swap_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
required_swap_kib="$((TEMP_SWAP_GIB * 1024 * 1024))"
swap_slack_kib="$((64 * 1024))"
if (( current_swap_kib + swap_slack_kib < required_swap_kib )); then
  if [[ "${ALLOW_TEMP_SWAP:-0}" != "1" ]]; then
    echo "SwapTotal is ${current_swap_kib} KiB; need about ${required_swap_kib} KiB plus ${swap_slack_kib} KiB slack. Re-run with ALLOW_TEMP_SWAP=1 to create temporary swap at ${SWAPFILE}." >&2
    exit 2
  fi
  run_privileged fallocate -l "${TEMP_SWAP_GIB}G" "${SWAPFILE}"
  run_privileged chmod 600 "${SWAPFILE}"
  run_privileged mkswap "${SWAPFILE}"
  run_privileged swapon "${SWAPFILE}"
  CREATED_SWAP=1
fi

{
  uname -a
  cat /etc/nv_tegra_release || true
  "${CUDA_HOME}/bin/nvcc" --version || true
} > "${LOG_DIR}/environment.txt" 2>&1

run_stage() {
  local name="$1"
  local command
  local stage_ld_library_path="${LD_LIBRARY_PATH}"
  command="$(stage_command "$name")"
  echo "== ${name} ==" | tee "${LOG_DIR}/${name}.status"
  if [[ "${FORCE_REBUILD_ENGINES:-0}" != "1" ]]; then
    if [[ "${name}" == "build-llm-engine" && -f "${ENGINE_DIR}/llm/llm.engine" ]]; then
      echo "cached: ${ENGINE_DIR}/llm/llm.engine" >> "${LOG_DIR}/${name}.status"
      return 0
    fi
    if [[ "${name}" == "build-audio-engine" && -f "${ENGINE_DIR}/audio/audio/audio_encoder.engine" ]]; then
      echo "cached: ${ENGINE_DIR}/audio/audio/audio_encoder.engine" >> "${LOG_DIR}/${name}.status"
      return 0
    fi
  fi
  if [[ "${name}" == "preprocess-audio" ]]; then
    stage_ld_library_path="${PYTORCH_LD_LIBRARY_PATH}"
  fi
  if [[ -x /usr/bin/time ]]; then
    env LD_LIBRARY_PATH="${stage_ld_library_path}" /usr/bin/time -v bash -lc "${command}" > "${LOG_DIR}/${name}.log" 2>&1
  else
    env LD_LIBRARY_PATH="${stage_ld_library_path}" bash -lc "${command}" > "${LOG_DIR}/${name}.log" 2>&1
  fi
}

run_stage build-llm-engine
run_stage build-audio-engine
run_stage preprocess-audio
run_stage benchmark-nano

echo "hosttrt logs: ${LOG_DIR}"
'''


_EDGELLM_BENCHMARK_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--strip-hypothesis-prefix-regex")
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
        scored_hypothesis = hypothesis
        if args.strip_hypothesis_prefix_regex:
            scored_hypothesis = re.sub(
                args.strip_hypothesis_prefix_regex,
                "",
                scored_hypothesis,
                flags=re.IGNORECASE,
            ).strip()
        duration_seconds = durations.get(sample_id, 0.0)
        rtf = latency / duration_seconds if duration_seconds > 0 else 0.0
        wer = _wer(reference, scored_hypothesis)
        row = {
            "sample_id": sample_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "scored_hypothesis": scored_hypothesis,
            "duration_seconds": duration_seconds,
            "latency_seconds": latency,
            "rtf": rtf,
            "wer": wer,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
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
