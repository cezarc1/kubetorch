from pathlib import Path

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
            "--copy-audio",
        ]
    )

    assert (tmp_path / "bundle/pipeline.json").exists()
    assert str(tmp_path / "bundle/pipeline.json") in capsys.readouterr().out
