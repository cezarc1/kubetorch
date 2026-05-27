from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from .constants import run_scoped_key
from .reporting import base_manifest, IssueLedger, publish_file, safe_note, write_json


def collect_environment() -> dict:
    packages = {}
    for name in (
        "kubetorch",
        "datasets",
        "huggingface-hub",
        "torch",
        "torchcodec",
        "open-clip-torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpu = {"torch_cuda_available": False}
    try:
        import torch

        gpu = {
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "torch_device_count": torch.cuda.device_count(),
            "torch_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        gpu["torch_error"] = str(exc)

    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "gpu": gpu,
        "environment": {
            key: os.getenv(key)
            for key in (
                "KT_RUN_ID",
                "KT_NAMESPACE",
                "KT_WORKDIR_KEY",
                "KT_LOGS_KEY",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_DATASETS_CACHE",
            )
        },
    }


def run_env_probe(*, output_dir: Path, namespace: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = IssueLedger(output_dir / "issues.md")
    manifest = base_manifest("env-probe") | collect_environment()
    target = write_json(output_dir / "environment.json", manifest)
    safe_note(
        "Environment probe completed: "
        f"cuda_available={manifest['gpu'].get('torch_cuda_available')}, "
        f"devices={manifest['gpu'].get('torch_devices')}"
    )
    publish_file(
        namespace=namespace,
        key=run_scoped_key("environment.json"),
        path=target,
        name="environment",
        metadata={"command": "env-probe"},
    )
    publish_file(
        namespace=namespace,
        key=run_scoped_key("issues.md"),
        path=ledger.path,
        name="issues",
        metadata={"command": "env-probe"},
    )
    return target
