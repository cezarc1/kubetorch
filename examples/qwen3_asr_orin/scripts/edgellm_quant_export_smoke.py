from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


EDGE_LLM_REPO = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
EDGE_LLM_REF = "v0.8.0"
EDGE_LLM_DEPS = (
    "onnx==1.19.0",
    "onnxscript==0.7.0",
    "onnx-graphsurgeon==0.6.1",
    "safetensors==0.7.0",
    "tiktoken==0.13.0",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded TensorRT-Edge-LLM Qwen3-ASR quant/export smoke.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("/tmp/qwen3-asr-model"))
    parser.add_argument("--no-snapshot-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3-asr-orin/edgellm-quant-export-smoke"))
    parser.add_argument("--quantization", default="int8_sq")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--edge-llm-dir", type=Path, default=Path("/tmp/TensorRT-Edge-LLM"))
    parser.add_argument("--edge-llm-ref", default=EDGE_LLM_REF)
    parser.add_argument("--quant-timeout-seconds", type=int, default=5400)
    parser.add_argument("--export-timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep-torchvision", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, object]] = []

    def record_step(name: str, command: list[str], *, timeout: int | None = None, required: bool = True) -> int:
        result = run_logged(name=name, command=command, log_dir=log_dir, timeout=timeout)
        steps.append(result)
        write_json(output_dir / "steps.json", steps)
        if required and result["returncode"] != 0:
            raise RuntimeError(f"{name} failed with exit code {result['returncode']}")
        return int(result["returncode"])

    exit_code = 1
    failure: str | None = None
    try:
        write_json(
            output_dir / "environment.json",
            {
                "edge_llm_ref": args.edge_llm_ref,
                "model_id": args.model_id,
                "num_samples": args.num_samples,
                "python": sys.version,
                "quantization": args.quantization,
                "run_id": os.getenv("KT_RUN_ID"),
            },
        )
        record_step("nvidia-smi", ["nvidia-smi"], required=False)
        if not args.keep_torchvision:
            record_step("pip-uninstall-torchvision", [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], required=False)
        if not args.skip_install:
            install_edge_llm(edge_llm_dir=args.edge_llm_dir, edge_llm_ref=args.edge_llm_ref, record_step=record_step)

        model_dir = args.model_id
        if not args.no_snapshot_model and not Path(args.model_id).exists():
            model_dir = snapshot_model(args.model_id, args.model_cache_dir, output_dir, steps)

        quantized_dir = output_dir / f"Qwen3-ASR-1.7B-{args.quantization}"
        onnx_dir = output_dir / f"Qwen3-ASR-1.7B-{args.quantization}-ONNX"
        record_step(
            "quantize",
            [
                "tensorrt-edgellm-quantize",
                "llm",
                "--model_dir",
                model_dir,
                "--output_dir",
                str(quantized_dir),
                "--quantization",
                args.quantization,
                "--num_samples",
                str(args.num_samples),
            ],
            timeout=args.quant_timeout_seconds,
        )
        if not args.skip_export:
            record_step(
                "export-onnx",
                ["tensorrt-edgellm-export", str(quantized_dir), str(onnx_dir)],
                timeout=args.export_timeout_seconds,
            )
        exit_code = 0
    except Exception as exc:
        failure = repr(exc)
        print(f"[smoke] failure: {failure}", file=sys.stderr)
    finally:
        summary = {
            "exit_code": exit_code,
            "failure": failure,
            "output_dir": str(output_dir),
            "steps": steps,
        }
        write_json(output_dir / "summary.json", summary)
        if args.upload:
            upload_result(output_dir)
    return exit_code


def install_edge_llm(edge_llm_dir: Path, edge_llm_ref: str, record_step) -> None:
    record_step("pip-install-edge-llm-deps", [sys.executable, "-m", "pip", "install", "--quiet", *EDGE_LLM_DEPS])
    if edge_llm_dir.exists():
        shutil.rmtree(edge_llm_dir)
    record_step(
        "clone-tensorrt-edge-llm",
        [
            "env",
            "GIT_LFS_SKIP_SMUDGE=1",
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            edge_llm_ref,
            EDGE_LLM_REPO,
            str(edge_llm_dir),
        ],
        timeout=900,
    )
    record_step("pip-install-edge-llm", [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(edge_llm_dir)])
    record_step("tensorrt-edgellm-quantize-help", ["tensorrt-edgellm-quantize", "--help"])


def snapshot_model(model_id: str, model_cache_dir: Path, output_dir: Path, steps: list[dict[str, object]]) -> str:
    started = time.monotonic()
    print(f"[smoke] snapshotting {model_id} to {model_cache_dir}", flush=True)
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_cache_dir),
            local_dir_use_symlinks=False,
        )
        result = {
            "command": ["huggingface_hub.snapshot_download", model_id],
            "duration_seconds": time.monotonic() - started,
            "log_path": None,
            "name": "snapshot-model",
            "returncode": 0,
            "tail": [f"snapshot_dir={path}"],
        }
        steps.append(result)
        write_json(output_dir / "steps.json", steps)
        print(f"[smoke] snapshot complete: {path}", flush=True)
        return str(path)
    except Exception as exc:
        result = {
            "command": ["huggingface_hub.snapshot_download", model_id],
            "duration_seconds": time.monotonic() - started,
            "log_path": None,
            "name": "snapshot-model",
            "returncode": 1,
            "tail": [repr(exc)],
        }
        steps.append(result)
        write_json(output_dir / "steps.json", steps)
        raise


def run_logged(
    *,
    name: str,
    command: list[str],
    log_dir: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    effective_command = ["timeout", f"{timeout}s", *command] if timeout is not None else command
    print(f"[smoke] running {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            effective_command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            lines.append(line.rstrip("\n"))
            if len(lines) > 80:
                lines.pop(0)
        returncode = process.wait()
        if returncode == 124 and timeout is not None:
            timeout_line = f"[smoke] {name} timed out after {timeout} seconds"
            print(timeout_line, flush=True)
            log_file.write(timeout_line + "\n")
            lines.append(timeout_line)
    elapsed = time.monotonic() - started
    result = {
        "command": command,
        "duration_seconds": elapsed,
        "log_path": str(log_path),
        "name": name,
        "returncode": returncode,
        "tail": lines[-40:],
    }
    print(f"[smoke] finished {name}: exit={returncode} duration={elapsed:.1f}s", flush=True)
    return result


def upload_result(output_dir: Path) -> None:
    run_id = os.getenv("KT_RUN_ID")
    namespace = os.getenv("KT_NAMESPACE", "kubetorch")
    if not run_id:
        print("[smoke] KT_RUN_ID is unset; skipping upload", flush=True)
        return
    try:
        import kubetorch as kt

        key = f"runs/{run_id}/artifacts/qwen3-asr-edgellm-smoke"
        artifact_filter = (
            "--include='*/' "
            "--include='*.json' "
            "--include='*.log' "
            "--include='*.onnx' "
            "--include='*.onnx.data' "
            "--include='*.safetensors' "
            "--exclude='*'"
        )
        kt.put(key=key, src=output_dir, namespace=namespace, force=True, filter_options=artifact_filter)
        kt.artifact(
            "qwen3-asr-edgellm-smoke",
            uri=f"kt://{namespace}/{key}",
            kind="kt-data-store",
            author="agent",
            metadata={"source": "examples/qwen3_asr_orin/scripts/edgellm_quant_export_smoke.py"},
        )
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        kt.note(
            f"TensorRT-Edge-LLM smoke exit={summary['exit_code']} failure={summary['failure']}",
            author="agent",
        )
        print(f"[smoke] uploaded result to kt://{namespace}/{key}", flush=True)
    except Exception as exc:
        print(f"[smoke] upload failed: {exc!r}", file=sys.stderr, flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
