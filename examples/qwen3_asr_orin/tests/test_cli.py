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


def test_cli_exports_fp16_trt_edgellm_bundle(tmp_path: Path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            AudioExample(
                id="cli-fp16-sample",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(audio_path),
                transcript="hello fp16",
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
            "/models/Qwen3-ASR-0.6B",
            "--qwen-asr-root",
            "/opt/Qwen3-ASR",
            "--format",
            "fp16",
            "--runtime-output-dir",
            "/work/bundle",
        ]
    )

    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    assert pipeline["quantization"]["format"] == "fp16"
    assert [stage["name"] for stage in pipeline["stages"]][0] == "export-onnx"


def test_cli_quantize_modelopt_defaults_to_qwen3_asr_06b(tmp_path: Path, capsys):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    main(
        [
            "quantize-modelopt",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "quant"),
            "--write-spec-only",
        ]
    )

    spec_path = tmp_path / "quant" / "quantization-spec.json"
    payload = json.loads(spec_path.read_text())
    assert payload["model_id"] == "Qwen/Qwen3-ASR-0.6B"
    assert str(spec_path) in capsys.readouterr().out


def test_cli_rescores_edgellm_results(tmp_path: Path, capsys):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "reference": "when you call someone",
                "hypothesis": "language EnglishWhen you call someone.",
                "duration_seconds": 4.0,
                "latency_seconds": 6.0,
                "returncode": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(
        [
            "rescore-edgellm-results",
            "--samples",
            str(samples_path),
            "--output-dir",
            str(tmp_path / "rescored"),
        ]
    )

    summary = json.loads((tmp_path / "rescored" / "summary.json").read_text())
    assert summary["mean_wer"] == 0.0
    assert str(tmp_path / "rescored" / "summary.json") in capsys.readouterr().out


def test_cli_compares_edgellm_runs(tmp_path: Path, capsys):
    first = tmp_path / "int8-summary.json"
    second = tmp_path / "fp16-summary.json"
    first.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "errors": 0,
                "total_audio_seconds": 10.0,
                "total_latency_seconds": 12.0,
                "aggregate_rtf": 1.2,
                "mean_latency_seconds": 6.0,
                "mean_rtf": 1.3,
                "mean_wer": 0.05,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "errors": 1,
                "total_audio_seconds": 10.0,
                "total_latency_seconds": 15.0,
                "aggregate_rtf": 1.5,
                "mean_latency_seconds": 7.5,
                "mean_rtf": 1.6,
                "mean_wer": 0.08,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.md"

    main(
        [
            "compare-edgellm-runs",
            "--run",
            f"int8={first}",
            "--run",
            f"fp16={second}",
            "--output",
            str(output),
        ]
    )

    report = output.read_text()
    assert "| int8 | 2 | 0 | 1.200 | 6.000 | 1.300 | 5.00% |" in report
    assert "| fp16 | 2 | 1 | 1.500 | 7.500 | 1.600 | 8.00% |" in report
    assert str(output) in capsys.readouterr().out
