import json
from pathlib import Path

import qwen3_asr_orin.quantize as quantize
from qwen3_asr_orin.datasets import AudioExample, write_manifest
from qwen3_asr_orin.quantize import (
    QuantizationSpec,
    int8_smoothquant_config,
    write_quantization_spec,
)


def test_int8_smoothquant_config_preserves_audio_and_lm_head_modules():
    config = int8_smoothquant_config()

    assert config["algorithm"] == "smoothquant"
    assert {"quantizer_name": "*weight_quantizer", "enable": True, "cfg": {"num_bits": 8, "axis": 0}} in config[
        "quant_cfg"
    ]
    assert {"quantizer_name": "*input_quantizer", "enable": True, "cfg": {"num_bits": 8, "axis": None}} in config[
        "quant_cfg"
    ]
    assert {"quantizer_name": "*audio_tower*", "enable": False} in config["quant_cfg"]
    assert {"quantizer_name": "*lm_head*", "enable": False} in config["quant_cfg"]


def test_write_quantization_spec_records_reproducible_inputs(tmp_path: Path):
    spec = QuantizationSpec(
        model_id="Qwen/Qwen3-ASR-1.7B",
        output_dir=str(tmp_path / "out"),
        manifest_path="/data/manifest.jsonl",
    )

    path = write_quantization_spec(spec, tmp_path / "out")

    payload = json.loads(path.read_text())
    assert payload["model_id"] == "Qwen/Qwen3-ASR-1.7B"
    assert payload["manifest_path"] == "/data/manifest.jsonl"
    assert payload["precision"] == "int8"
    assert payload["modelopt_config"]["algorithm"] == "smoothquant"


def test_export_trt_edgellm_bundle_materializes_audio_prompts_and_pipeline(tmp_path: Path):
    audio_dir = tmp_path / "source-audio"
    audio_dir.mkdir()
    first_audio = audio_dir / "sample-one.wav"
    second_audio = audio_dir / "sample-two.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            AudioExample(
                id="sample-one",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(first_audio),
                transcript="hello orin",
                duration_seconds=1.25,
            ),
            AudioExample(
                id="sample-two",
                dataset="fleurs",
                language="fr",
                split="validation",
                audio_path=str(second_audio),
                transcript="bonjour orin",
                duration_seconds=2.5,
            ),
        ],
        manifest_path,
    )

    bundle = quantize.export_trt_edgellm_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "bundle",
        model_path="/models/Qwen3-ASR-1.7B",
        qwen_asr_root="/opt/Qwen3-ASR",
        quant_format="int8",
        materialize_audio="copy",
    )

    assert bundle.sample_count == 2
    assert (tmp_path / "bundle/audio/sample-one.wav").read_bytes() == b"first"
    assert (tmp_path / "bundle/audio/sample-two.wav").read_bytes() == b"second"
    assert (tmp_path / "bundle/prompts.txt").read_text() == "sample-one\thello orin\nsample-two\tbonjour orin\n"
    benchmark_script = tmp_path / "bundle/scripts/qwen3_asr_edgellm_benchmark.py"
    assert benchmark_script.exists()
    compile(benchmark_script.read_text(), str(benchmark_script), "exec")

    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    assert pipeline["quantization"]["format"] == "int8"
    assert pipeline["quantization"]["model_path"] == "/models/Qwen3-ASR-1.7B"
    assert pipeline["quantization"]["qwen_asr_root"] == "/opt/Qwen3-ASR"
    assert pipeline["quantization"]["tensorrt_edgellm_version"] == "v0.8.0"
    assert "tensorrt-edgellm-quantize llm" in pipeline["stages"][0]["command"]
    assert "--quantization int8_sq" in pipeline["stages"][0]["command"]
    assert "tensorrt-edgellm-export" in pipeline["stages"][1]["command"]
    assert "llm_build" in pipeline["stages"][2]["command"]
    assert "audio_build" in pipeline["stages"][3]["command"]
    assert "tensorrt-edgellm-preprocess-audio" in pipeline["stages"][4]["command"]
    assert "scripts/qwen3_asr_edgellm_benchmark.py" in pipeline["stages"][-1]["command"]
    assert "--manifest" in pipeline["stages"][-1]["command"]
    assert "llm_inference" in pipeline["stages"][-1]["command"]
