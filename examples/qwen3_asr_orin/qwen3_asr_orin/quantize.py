from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantizationSpec:
    model_id: str
    output_dir: str
    manifest_path: str
    algorithm: str = "smoothquant"
    precision: str = "int8"
    quantized_modules: str = "llm_decoder"
    preserved_modules: tuple[str, ...] = ("*audio*", "*audio_tower*", "*lm_head*")


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
