from __future__ import annotations

import argparse
from pathlib import Path

from qwen3_asr_orin.benchmark import run_benchmark
from qwen3_asr_orin.datasets import prepare_public_manifest
from qwen3_asr_orin.quantize import QuantizationSpec, run_modelopt_quantization, write_quantization_spec


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


if __name__ == "__main__":
    main()
