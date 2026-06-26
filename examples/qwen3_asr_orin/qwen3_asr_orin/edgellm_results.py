from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from qwen3_asr_orin.metrics import word_error_rate

DEFAULT_LANGUAGE_PREFIX_REGEX = (
    r"^language\s+(?:English|German|Deutsch|Spanish|Español|French|Français)\s*"
)


def strip_hypothesis_prefix(
    text: str, pattern: str = DEFAULT_LANGUAGE_PREFIX_REGEX
) -> str:
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def rescore_samples_file(
    samples_path: Path,
    output_dir: Path,
    strip_prefix_regex: str = DEFAULT_LANGUAGE_PREFIX_REGEX,
) -> dict:
    rows = list(_read_jsonl(samples_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    rescored_rows = []
    for row in rows:
        rescored = dict(row)
        rescored["scored_hypothesis"] = strip_hypothesis_prefix(
            str(row.get("hypothesis", "")), strip_prefix_regex
        )
        rescored["wer"] = word_error_rate(
            str(row.get("reference", "")), rescored["scored_hypothesis"]
        )
        rescored["rtf"] = _rtf(rescored)
        rescored_rows.append(rescored)

    rescored_samples_path = output_dir / "samples.jsonl"
    _write_jsonl(rescored_samples_path, rescored_rows)
    summary = _summary(
        rescored_rows,
        samples_path=samples_path,
        rescored_samples_path=rescored_samples_path,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rtf(row: dict) -> float:
    duration_seconds = float(row.get("duration_seconds", 0.0))
    if duration_seconds <= 0:
        return 0.0
    return float(row.get("latency_seconds", 0.0)) / duration_seconds


def _summary(rows: list[dict], samples_path: Path, rescored_samples_path: Path) -> dict:
    total_audio_seconds = sum(float(row.get("duration_seconds", 0.0)) for row in rows)
    total_latency_seconds = sum(float(row.get("latency_seconds", 0.0)) for row in rows)
    total_rtf = sum(float(row.get("rtf", 0.0)) for row in rows)
    total_wer = sum(float(row.get("wer", 0.0)) for row in rows)
    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "errors": sum(1 for row in rows if int(row.get("returncode", 0)) != 0),
        "total_audio_seconds": total_audio_seconds,
        "total_latency_seconds": total_latency_seconds,
        "mean_latency_seconds": total_latency_seconds / sample_count
        if sample_count
        else 0.0,
        "mean_rtf": total_rtf / sample_count if sample_count else 0.0,
        "aggregate_rtf": total_latency_seconds / total_audio_seconds
        if total_audio_seconds > 0
        else 0.0,
        "mean_wer": total_wer / sample_count if sample_count else 0.0,
        "rescored_from": str(samples_path),
        "samples_path": str(rescored_samples_path),
    }
