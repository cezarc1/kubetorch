from __future__ import annotations

import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from qwen3_asr_orin.client import TranscriptionClient
from qwen3_asr_orin.datasets import AudioExample, load_manifest
from qwen3_asr_orin.metrics import word_error_rate


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    dataset: str
    language: str
    duration_seconds: float
    latency_seconds: float
    transcript: str
    reference: str
    error: str | None


def summarize_samples(
    samples: Iterable[BenchmarkSample],
    wall_seconds: float,
    model: str,
    runtime: str,
    precision: str,
) -> dict:
    sample_list = list(samples)
    completed = [sample for sample in sample_list if sample.error is None]
    errored = [sample for sample in sample_list if sample.error is not None]
    latencies = [sample.latency_seconds for sample in completed]
    rtfs = [sample.latency_seconds / sample.duration_seconds for sample in completed]
    wers = [word_error_rate(sample.reference, sample.transcript) for sample in completed]
    audio_seconds_completed = sum(sample.duration_seconds for sample in completed)

    return {
        "model": model,
        "runtime": runtime,
        "precision": precision,
        "attempted": len(sample_list),
        "completed": len(completed),
        "errors": {
            "total": len(errored),
            "samples": [sample.error for sample in errored[:5]],
        },
        "wall_seconds": wall_seconds,
        "audio_seconds_completed": audio_seconds_completed,
        "completed_requests_per_second": len(completed) / wall_seconds if wall_seconds else 0.0,
        "throughput_audio_seconds_per_second": audio_seconds_completed / wall_seconds if wall_seconds else 0.0,
        "latency_seconds": _percentiles(latencies),
        "rtf": {
            "mean": statistics.fmean(rtfs) if rtfs else 0.0,
            **_percentiles(rtfs),
        },
        "wer": {
            "mean": statistics.fmean(wers) if wers else 0.0,
            **_percentiles(wers),
        },
        "by_dataset": _group_counts(completed, lambda sample: sample.dataset),
        "by_language": _group_counts(completed, lambda sample: sample.language),
    }


def run_benchmark(
    manifest_path: Path,
    output_dir: Path,
    base_url: str,
    model: str = "qwen3-asr",
    runtime: str = "sglang",
    precision: str = "fp16",
    concurrency: int = 1,
    limit: int | None = None,
    timeout_seconds: float = 120.0,
) -> dict:
    examples = load_manifest(manifest_path)
    if limit is not None:
        examples = examples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    client = TranscriptionClient(base_url=base_url, model=model, timeout_seconds=timeout_seconds)
    started = time.perf_counter()
    samples = _transcribe_examples(client, examples, concurrency=concurrency)
    wall_seconds = time.perf_counter() - started
    summary = summarize_samples(samples, wall_seconds=wall_seconds, model=model, runtime=runtime, precision=precision)

    _write_jsonl(output_dir / "samples.jsonl", (asdict(sample) for sample in samples))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(runtime=runtime, precision=precision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _transcribe_examples(
    client: TranscriptionClient,
    examples: list[AudioExample],
    concurrency: int,
) -> list[BenchmarkSample]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_transcribe_one, client, example): example.id for example in examples}
        samples = [future.result() for future in as_completed(futures)]
    return sorted(samples, key=lambda sample: sample.id)


def _transcribe_one(client: TranscriptionClient, example: AudioExample) -> BenchmarkSample:
    started = time.perf_counter()
    transcript = ""
    error = None
    try:
        transcript = client.transcribe(Path(example.audio_path)).text
    except Exception as exc:
        error = str(exc)
    latency_seconds = time.perf_counter() - started
    return BenchmarkSample(
        id=example.id,
        dataset=example.dataset,
        language=example.language,
        duration_seconds=example.duration_seconds,
        latency_seconds=latency_seconds,
        transcript=transcript,
        reference=example.transcript,
        error=error,
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _group_counts(samples: Iterable[BenchmarkSample], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = key_fn(sample)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")


def _environment(runtime: str, precision: str) -> dict:
    return {
        "runtime": runtime,
        "precision": precision,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
