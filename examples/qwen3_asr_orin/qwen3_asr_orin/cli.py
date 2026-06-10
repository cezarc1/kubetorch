from __future__ import annotations

import argparse
from pathlib import Path

from qwen3_asr_orin.benchmark import run_benchmark
from qwen3_asr_orin.datasets import prepare_public_manifest
from qwen3_asr_orin.edgellm_results import DEFAULT_LANGUAGE_PREFIX_REGEX, rescore_samples_file
from qwen3_asr_orin.quantize import (
    QuantizationSpec,
    export_trt_edgellm_bundle,
    run_modelopt_quantization,
    write_quantization_spec,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Orin benchmark helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-manifest", help="Download a small LibriSpeech + FLEURS manifest")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--librispeech-count", type=int, default=8)
    prepare.add_argument("--fleurs-count-per-language", type=int, default=2)
    prepare.add_argument("--fleurs-languages", default="en_us,es_419,fr_fr,de_de")

    bench = subparsers.add_parser("bench", help="Benchmark an OpenAI-compatible audio transcription endpoint")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--output-dir", type=Path, required=True)
    bench.add_argument("--base-url", required=True)
    bench.add_argument("--model", default="qwen3-asr")
    bench.add_argument("--runtime", default="sglang")
    bench.add_argument("--precision", default="fp16")
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--limit", type=int)
    bench.add_argument("--timeout-seconds", type=float, default=120.0)

    quant = subparsers.add_parser("quantize-modelopt", help="Write or run the ModelOpt INT8 quantization spec")
    quant.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    quant.add_argument("--manifest", type=Path, required=True)
    quant.add_argument("--output-dir", type=Path, required=True)
    quant.add_argument("--write-spec-only", action="store_true")

    trt_bundle = subparsers.add_parser(
        "export-trt-edgellm-bundle",
        help="Write TensorRT-Edge-LLM calibration, benchmark, and pipeline inputs",
    )
    trt_bundle.add_argument("--manifest", type=Path, required=True)
    trt_bundle.add_argument("--output-dir", type=Path, required=True)
    trt_bundle.add_argument("--model-path", required=True)
    trt_bundle.add_argument("--qwen-asr-root", required=True)
    trt_bundle.add_argument("--format", choices=("int8", "int4"), default="int8")
    trt_bundle.add_argument("--trt-edgellm-root", default="$TRT_EDGELLM_ROOT")
    trt_bundle.add_argument("--jetson-node-name", default="jetson-orin-nano-01")
    trt_bundle.add_argument("--host-cuda-path", default="/usr/local/cuda-13.2")
    trt_bundle.add_argument("--host-usr-path", default="/usr")
    trt_bundle.add_argument("--temp-swap-gib", type=int, default=16)
    trt_bundle.add_argument("--runtime-output-dir")
    trt_bundle.add_argument("--copy-audio", action="store_true")

    rescore = subparsers.add_parser("rescore-edgellm-results", help="Recompute WER for saved Edge-LLM samples")
    rescore.add_argument("--samples", type=Path, required=True)
    rescore.add_argument("--output-dir", type=Path, required=True)
    rescore.add_argument("--strip-prefix-regex", default=DEFAULT_LANGUAGE_PREFIX_REGEX)

    args = parser.parse_args(argv)

    if args.command == "prepare-manifest":
        languages = tuple(language.strip() for language in args.fleurs_languages.split(",") if language.strip())
        manifest = prepare_public_manifest(
            output_dir=args.output_dir,
            librispeech_count=args.librispeech_count,
            fleurs_count_per_language=args.fleurs_count_per_language,
            fleurs_languages=languages,
        )
        print(manifest)
    elif args.command == "bench":
        summary = run_benchmark(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            base_url=args.base_url,
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            concurrency=args.concurrency,
            limit=args.limit,
            timeout_seconds=args.timeout_seconds,
        )
        print(summary)
    elif args.command == "quantize-modelopt":
        spec = QuantizationSpec(
            model_id=args.model_id,
            output_dir=str(args.output_dir),
            manifest_path=str(args.manifest),
        )
        if args.write_spec_only:
            print(write_quantization_spec(spec, args.output_dir))
        else:
            run_modelopt_quantization(spec)
    elif args.command == "export-trt-edgellm-bundle":
        bundle = export_trt_edgellm_bundle(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            model_path=args.model_path,
            qwen_asr_root=args.qwen_asr_root,
            quant_format=args.format,
            materialize_audio="copy" if args.copy_audio else "symlink",
            trt_edgellm_root=args.trt_edgellm_root,
            jetson_node_name=args.jetson_node_name,
            host_cuda_path=args.host_cuda_path,
            host_usr_path=args.host_usr_path,
            temp_swap_gib=args.temp_swap_gib,
            runtime_output_dir=args.runtime_output_dir,
        )
        print(bundle.pipeline_path)
    elif args.command == "rescore-edgellm-results":
        rescore_samples_file(
            samples_path=args.samples,
            output_dir=args.output_dir,
            strip_prefix_regex=args.strip_prefix_regex,
        )
        print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
