import json
from pathlib import Path

from qwen3_asr_orin.quantize import QuantizationSpec, int8_smoothquant_config, write_quantization_spec


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
