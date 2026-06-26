from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    script_path = (
        Path(__file__).parents[1] / "scripts" / "edgellm_quant_export_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "edgellm_quant_export_smoke", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_prefix_uses_model_id_and_quantization():
    smoke = _load_smoke_module()

    assert (
        smoke.artifact_prefix("Qwen/Qwen3-ASR-0.6B", "int8_sq")
        == "Qwen3-ASR-0.6B-int8_sq"
    )
    assert (
        smoke.artifact_prefix("/models/Qwen3-ASR-1.7B", "int4_awq")
        == "Qwen3-ASR-1.7B-int4_awq"
    )


def test_upload_filter_keeps_complete_export_tree():
    smoke = _load_smoke_module()

    filters = smoke.artifact_filter_options()

    assert "--include='*.onnx'" in filters
    assert "--include='*.onnx.data'" in filters
    assert "--include='*.safetensors'" in filters
    assert "--exclude='*'" in filters


def test_artifact_upload_filters_bypass_repo_gitignore(monkeypatch):
    smoke = _load_smoke_module()
    monkeypatch.setenv("KT_RSYNC_FILTERS", "sentinel")

    with smoke.artifact_upload_filters():
        assert "*.onnx" not in smoke.artifact_rsync_exclude_override()
        assert (
            smoke.os.environ["KT_RSYNC_FILTERS"]
            == smoke.artifact_rsync_exclude_override()
        )

    assert smoke.os.environ["KT_RSYNC_FILTERS"] == "sentinel"
