import json
from pathlib import Path

import qwen3_asr_orin.quantize as quantize
from qwen3_asr_orin.datasets import AudioExample, load_manifest, write_manifest
from qwen3_asr_orin.quantize import (
    int8_smoothquant_config,
    QuantizationSpec,
    write_quantization_spec,
)


def test_int8_smoothquant_config_preserves_audio_and_lm_head_modules():
    config = int8_smoothquant_config()

    assert config["algorithm"] == "smoothquant"
    assert {
        "quantizer_name": "*weight_quantizer",
        "enable": True,
        "cfg": {"num_bits": 8, "axis": 0},
    } in config["quant_cfg"]
    assert {
        "quantizer_name": "*input_quantizer",
        "enable": True,
        "cfg": {"num_bits": 8, "axis": None},
    } in config["quant_cfg"]
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


def test_export_trt_edgellm_bundle_materializes_audio_prompts_and_pipeline(
    tmp_path: Path,
):
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
        runtime_output_dir="/work/bundle",
    )

    assert bundle.sample_count == 2
    assert (tmp_path / "bundle/audio/sample-one.wav").read_bytes() == b"first"
    assert (tmp_path / "bundle/audio/sample-two.wav").read_bytes() == b"second"
    bundled_manifest = load_manifest(tmp_path / "bundle/manifest.jsonl")
    assert bundled_manifest[0].audio_path == "/work/bundle/audio/sample-one.wav"
    assert bundled_manifest[1].audio_path == "/work/bundle/audio/sample-two.wav"
    assert (
        tmp_path / "bundle/prompts.txt"
    ).read_text() == "sample-one\thello orin\nsample-two\tbonjour orin\n"
    benchmark_script = tmp_path / "bundle/scripts/qwen3_asr_edgellm_benchmark.py"
    hosttrt_runner = tmp_path / "bundle/scripts/run_jetson_hosttrt_pipeline.sh"
    assert benchmark_script.exists()
    benchmark_source = benchmark_script.read_text()
    compile(benchmark_source, str(benchmark_script), "exec")
    assert "phase_events" in benchmark_source
    assert "--limit" in benchmark_source
    assert "--sample-id" in benchmark_source
    benchmark_namespace: dict[str, object] = {"__name__": "generated_benchmark_test"}
    exec(benchmark_source, benchmark_namespace)
    assert (
        benchmark_namespace["_first_text"](
            {
                "input_file": "/tmp/input.json",
                "responses": [{"output_text": "language Englishhello orin"}],
            }
        )
        == "language Englishhello orin"
    )
    assert (
        benchmark_namespace["_strip_hypothesis_prefix"](
            "language EnglishWhen you call someone who is thousands of miles away.",
            r"^language\s+(?:English|German|Deutsch|Spanish|Español|French|Français)\s*",
        )
        == "When you call someone who is thousands of miles away."
    )
    assert (
        benchmark_namespace["_strip_hypothesis_prefix"](
            "language EnglishLaut Polizei wird der Fahrer nicht angeklagt.",
            r"^language\s+(?:English|German|Deutsch|Spanish|Español|French|Français)\s*",
        )
        == "Laut Polizei wird der Fahrer nicht angeklagt."
    )
    assert hosttrt_runner.exists()
    hosttrt_runner_text = hosttrt_runner.read_text()
    assert "ALLOW_TEMP_SWAP=1" in hosttrt_runner_text
    assert "swap_slack_kib" in hosttrt_runner_text
    assert "[[ -x /usr/bin/time ]]" in hosttrt_runner_text
    assert "EDGELLM_PLUGIN_PATH" in hosttrt_runner_text
    assert "NvInfer_edgellm_plugin" in hosttrt_runner_text
    assert "/usr/local/cuda-13.0/targets/sbsa-linux/lib" in hosttrt_runner_text
    assert "PYTORCH_LD_LIBRARY_PATH" in hosttrt_runner_text
    assert 'if [[ "${name}" == "preprocess-audio" ]]' in hosttrt_runner_text
    assert "FORCE_REBUILD_ENGINES" in hosttrt_runner_text
    assert "cached: ${ENGINE_DIR}/llm/llm.engine" in hosttrt_runner_text
    assert "needs_temp_swap" in hosttrt_runner_text
    assert "Engine cache is complete; skipping swap guard." in hosttrt_runner_text
    assert "NSYS_PROFILE_BENCHMARK" in hosttrt_runner_text
    assert "BENCHMARK_LIMIT" in hosttrt_runner_text
    assert "tegrastats" in hosttrt_runner_text
    assert "NSYS_BIN" in hosttrt_runner_text
    assert "/usr/local/cuda/bin/nsys" in hosttrt_runner_text
    assert "nsys profile" in hosttrt_runner_text
    assert "nsys stats" in hosttrt_runner_text

    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    assert pipeline["quantization"]["format"] == "int8"
    assert pipeline["quantization"]["model_path"] == "/models/Qwen3-ASR-1.7B"
    assert pipeline["quantization"]["qwen_asr_root"] == "/opt/Qwen3-ASR"
    assert pipeline["quantization"]["tensorrt_edgellm_version"] == "v0.8.0"
    assert pipeline["quantization"]["local_output_dir"] == str(tmp_path / "bundle")
    assert pipeline["quantization"]["runtime_output_dir"] == "/work/bundle"
    assert pipeline["runtime_requirements"]["benchmark_node"] == "jetson-orin-nano-01"
    assert pipeline["runtime_requirements"]["cuda_home"] == "/usr/local/cuda"
    assert pipeline["runtime_requirements"]["tensorrt_version"] == "10.16.2.10"
    assert pipeline["runtime_requirements"]["temp_swap_gib"] == 16
    assert pipeline["runtime_requirements"]["kubernetes"]["runtime_class_name"] is None
    assert (
        pipeline["runner"]["jetson_hosttrt"]
        == "/work/bundle/scripts/run_jetson_hosttrt_pipeline.sh"
    )
    assert pipeline["runner"]["local_jetson_hosttrt"].endswith(
        "run_jetson_hosttrt_pipeline.sh"
    )
    assert "tensorrt-edgellm-quantize llm" in pipeline["stages"][0]["command"]
    assert "--quantization int8_sq" in pipeline["stages"][0]["command"]
    assert "tensorrt-edgellm-export" in pipeline["stages"][1]["command"]
    assert "llm_build" in pipeline["stages"][2]["command"]
    assert "audio_build" in pipeline["stages"][3]["command"]
    assert "--minTimeSteps 100 " in pipeline["stages"][3]["command"]
    assert "tensorrt-edgellm-preprocess-audio" in pipeline["stages"][4]["command"]
    assert "/work/bundle/audio" in pipeline["stages"][4]["command"]
    assert (
        "/work/bundle/scripts/qwen3_asr_edgellm_benchmark.py"
        in pipeline["stages"][-1]["command"]
    )
    assert "--manifest" in pipeline["stages"][-1]["command"]
    assert "llm_inference" in pipeline["stages"][-1]["command"]
    assert "--strip-hypothesis-prefix-regex" in pipeline["stages"][-1]["command"]
    assert "English|German|Deutsch|Spanish" in pipeline["stages"][-1]["command"]
    assert "--phase-events" in pipeline["stages"][-1]["command"]


def test_export_trt_edgellm_bundle_uses_model_specific_artifact_names(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            AudioExample(
                id="sample",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(audio_path),
                transcript="hello",
                duration_seconds=1.0,
            )
        ],
        manifest_path,
    )

    quantize.export_trt_edgellm_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "bundle",
        model_path="Qwen/Qwen3-ASR-0.6B",
        qwen_asr_root="/opt/Qwen3-ASR",
        quant_format="int8",
        runtime_output_dir="/work/bundle",
    )

    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    assert pipeline["quantization"]["model_name"] == "Qwen3-ASR-0.6B"
    assert (
        pipeline["quantization"]["quantized_dir"] == "/work/bundle/Qwen3-ASR-0.6B-int8"
    )
    assert (
        pipeline["quantization"]["onnx_dir"] == "/work/bundle/Qwen3-ASR-0.6B-int8-ONNX"
    )
    assert (
        pipeline["quantization"]["engine_dir"]
        == "/work/bundle/Qwen3-ASR-0.6B-int8-Engines"
    )
    assert pipeline["stages"][0]["command"].startswith(
        "tensorrt-edgellm-quantize llm --model_dir Qwen/Qwen3-ASR-0.6B "
    )


def test_export_trt_edgellm_bundle_can_generate_fp16_pipeline(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            AudioExample(
                id="sample",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(audio_path),
                transcript="hello",
                duration_seconds=1.0,
            )
        ],
        manifest_path,
    )

    quantize.export_trt_edgellm_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "bundle",
        model_path="/models/Qwen3-ASR-0.6B",
        qwen_asr_root="/opt/Qwen3-ASR",
        quant_format="fp16",
        runtime_output_dir="/work/bundle",
    )

    pipeline = json.loads((tmp_path / "bundle/pipeline.json").read_text())
    stage_names = [stage["name"] for stage in pipeline["stages"]]
    assert pipeline["quantization"]["format"] == "fp16"
    assert pipeline["quantization"]["method"] == "fp16"
    assert (
        pipeline["quantization"]["onnx_dir"] == "/work/bundle/Qwen3-ASR-0.6B-fp16-ONNX"
    )
    assert (
        pipeline["quantization"]["engine_dir"]
        == "/work/bundle/Qwen3-ASR-0.6B-fp16-Engines"
    )
    assert "quantize" not in stage_names
    assert stage_names[0] == "export-onnx"
    assert pipeline["stages"][0]["command"] == (
        "tensorrt-edgellm-export /models/Qwen3-ASR-0.6B /work/bundle/Qwen3-ASR-0.6B-fp16-ONNX"
    )
