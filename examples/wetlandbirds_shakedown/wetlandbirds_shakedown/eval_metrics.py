from __future__ import annotations

from collections import Counter
from typing import Any


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def compute_classification_metrics(
    predictions: list[dict[str, Any]],
    *,
    samples_requested: int,
) -> dict[str, Any]:
    evaluated = [
        prediction
        for prediction in predictions
        if prediction.get("crop_available") and prediction.get("ranked_species")
    ]
    true_labels = [str(prediction["true_species"]) for prediction in evaluated]
    pred_labels = [str(prediction["ranked_species"][0]) for prediction in evaluated]

    samples_evaluated = len(evaluated)
    crop_success_rate = (
        samples_evaluated / samples_requested if samples_requested else 0.0
    )
    top1_correct = sum(true == pred for true, pred in zip(true_labels, pred_labels))
    top3_correct = sum(
        true in [str(label) for label in prediction.get("ranked_species", [])[:3]]
        for true, prediction in zip(true_labels, evaluated)
    )

    per_class_counts = Counter(true_labels)
    per_class_recall = {}
    f1_scores = []
    for label in sorted(per_class_counts):
        tp = sum(
            true == label and pred == label
            for true, pred in zip(true_labels, pred_labels)
        )
        fp = sum(
            true != label and pred == label
            for true, pred in zip(true_labels, pred_labels)
        )
        fn = sum(
            true == label and pred != label
            for true, pred in zip(true_labels, pred_labels)
        )
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_class_recall[label] = recall
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "samples_requested": samples_requested,
        "samples_seen": len(predictions),
        "samples_evaluated": samples_evaluated,
        "crop_success_rate": crop_success_rate,
        "top1_accuracy": top1_correct / samples_evaluated if samples_evaluated else 0.0,
        "top3_accuracy": top3_correct / samples_evaluated if samples_evaluated else 0.0,
        "macro_f1": macro_f1,
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "per_class_recall": per_class_recall,
    }


def summarize_performance(
    *,
    samples_evaluated: int,
    timings: dict[str, float],
    device_info: dict[str, Any],
) -> dict[str, Any]:
    inference_seconds = timings.get("inference_seconds_total", 0.0)
    samples_per_second = (
        samples_evaluated / inference_seconds if inference_seconds else 0.0
    )
    return {
        **{key: _rounded(value) for key, value in timings.items()},
        "samples_per_second": _rounded(samples_per_second),
        **device_info,
    }
