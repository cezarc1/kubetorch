from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdgeLlmRun:
    label: str
    summary_path: Path
    sample_count: int
    errors: int
    aggregate_rtf: float
    mean_latency_seconds: float
    mean_rtf: float
    mean_wer: float


def parse_labeled_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"run must use label=path format: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"run label must not be empty: {value}")
    return label, Path(raw_path)


def load_edgellm_run(label: str, summary_path: Path) -> EdgeLlmRun:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return EdgeLlmRun(
        label=label,
        summary_path=summary_path,
        sample_count=int(payload.get("sample_count", 0)),
        errors=int(payload.get("errors", 0)),
        aggregate_rtf=float(payload.get("aggregate_rtf", 0.0)),
        mean_latency_seconds=float(payload.get("mean_latency_seconds", 0.0)),
        mean_rtf=float(payload.get("mean_rtf", 0.0)),
        mean_wer=float(payload.get("mean_wer", 0.0)),
    )


def render_comparison_markdown(runs: list[EdgeLlmRun]) -> str:
    lines = [
        "# Qwen3-ASR Orin TensorRT Comparison",
        "",
        "| Run | Samples | Errors | Aggregate RTF | Mean Latency (s) | Mean RTF | Mean WER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| "
            f"{run.label} | "
            f"{run.sample_count} | "
            f"{run.errors} | "
            f"{run.aggregate_rtf:.3f} | "
            f"{run.mean_latency_seconds:.3f} | "
            f"{run.mean_rtf:.3f} | "
            f"{run.mean_wer:.2%} |"
        )
    lines.extend(
        [
            "",
            "Lower RTF is faster; lower WER is better. Treat this as a compact index over saved summaries, not a replacement for profiler traces.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_report(labeled_runs: list[str], output_path: Path) -> Path:
    runs = [load_edgellm_run(*parse_labeled_summary(value)) for value in labeled_runs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_comparison_markdown(runs), encoding="utf-8")
    return output_path
