from pathlib import Path
import json

from qwen3_asr_orin.cli import main
from qwen3_asr_orin.datasets import AudioExample, write_manifest


def test_cli_exports_trt_edgellm_bundle(tmp_path: Path, capsys):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            AudioExample(
                id="cli-sample",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(audio_path),
                transcript="hello cli",
                duration_seconds=1.0,
            )
        ],
        manifest_path,
    )

    main(
        [
            "export-trt-edgellm-bundle",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--model-path",
            "/models/Qwen3-ASR-1.7B",
            "--qwen-asr-root",
            "/opt/Qwen3-ASR",
            "--host-cuda-path",
            "/host/usr/local/cuda-13.2",
            "--host-usr-path",
            "/host/usr",
            "--temp-swap-gib",
            "12",
            "--runtime-output-dir",
            "/work/bundle",
            "--copy-audio",
        ]
    )

    assert (tmp_path / "bundle/pipeline.json").exists()
    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    assert pipeline["runtime_requirements"]["cuda_home"] == "/host/usr/local/cuda-13.2"
    assert pipeline["runtime_requirements"]["host_usr_path"] == "/host/usr"
    assert pipeline["runtime_requirements"]["temp_swap_gib"] == 12
    assert pipeline["quantization"]["runtime_output_dir"] == "/work/bundle"
    assert pipeline["runner"]["usage"].startswith("ALLOW_TEMP_SWAP=1 /work/bundle/")
    assert str(tmp_path / "bundle/pipeline.json") in capsys.readouterr().out
