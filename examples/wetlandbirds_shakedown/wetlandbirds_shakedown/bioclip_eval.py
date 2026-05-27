from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path
from typing import Any

from .bioclip_smoke import _crop_from_video
from .constants import DATASET_ID, run_scoped_key
from .datasets_io import (
    choose_split,
    disable_video_decoding,
    load_dataset_kwargs,
    manifest_dataset_kwargs,
)
from .eval_metrics import compute_classification_metrics, summarize_performance
from .reporting import base_manifest, IssueLedger, publish_file, safe_note, write_json


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target


def _load_model_bundle(model_name: str) -> dict[str, Any]:
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return {
        "model_name": model_name,
        "model": model,
        "preprocess": preprocess_val,
        "tokenizer": tokenizer,
        "device": device,
        "torch": torch,
    }


def _device_info(model_bundle: dict[str, Any]) -> dict[str, Any]:
    if "device_info" in model_bundle:
        return model_bundle["device_info"]

    torch = model_bundle["torch"]
    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if cuda_available else None
    )
    return {
        "device": model_bundle["device"],
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "cuda_peak_memory_mb": cuda_peak_memory_mb,
    }


def _score_crop_species(
    model_bundle: dict[str, Any],
    crop: Any,
    candidate_species: list[str],
) -> list[dict[str, Any]]:
    if not candidate_species:
        return []

    torch = model_bundle["torch"]
    model = model_bundle["model"]
    preprocess = model_bundle["preprocess"]
    tokenizer = model_bundle["tokenizer"]
    device = model_bundle["device"]
    prompts = [f"a photo of a {species}" for species in candidate_species]

    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image = preprocess(crop).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()

    ranked = sorted(
        zip(candidate_species, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"species": species, "score": float(score)} for species, score in ranked]


def _load_dataset_with_split_fallback(
    load_dataset, kwargs: dict[str, Any]
) -> tuple[Any, str | None, str | None]:
    requested_split = kwargs.get("split")
    try:
        return load_dataset(**kwargs), requested_split, None
    except ValueError as exc:
        if not requested_split or "Bad split" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["split"] = "train"
        return load_dataset(**fallback_kwargs), "train", str(exc)


def run_bioclip_eval_smoke(
    *,
    output_dir: Path,
    namespace: str,
    split: str | None,
    sample_rows: int,
    model_name: str,
    streaming: bool = True,
) -> Path:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "hf-cache"
    ledger = IssueLedger(output_dir / "issues.md")
    timings = {
        "dataset_load_seconds": 0.0,
        "model_load_seconds": 0.0,
        "crop_decode_seconds_total": 0.0,
        "inference_seconds_total": 0.0,
    }

    safe_note(f"Starting BioCLIP eval smoke for {DATASET_ID} split={split}")

    kwargs = load_dataset_kwargs(cache_dir=cache_dir, split=split, streaming=streaming)
    start = time.perf_counter()
    dataset, effective_split, split_fallback_reason = _load_dataset_with_split_fallback(
        load_dataset, kwargs
    )
    dataset = disable_video_decoding(dataset)
    selected_split = choose_split(dataset, preferred=effective_split or split or "test")
    rows = list(islice(iter(selected_split), sample_rows))
    timings["dataset_load_seconds"] = time.perf_counter() - start
    if split_fallback_reason:
        ledger.add(
            category="dataset",
            severity="medium",
            summary="requested Hugging Face split was unavailable; fell back to train",
            evidence=split_fallback_reason,
            workaround="use the official Zenodo splits.json for paper-quality test-set evaluation",
        )

    candidate_species = sorted(
        {str(row["species"]) for row in rows if row.get("species") is not None}
    )

    start = time.perf_counter()
    model_bundle = _load_model_bundle(model_name)
    timings["model_load_seconds"] = time.perf_counter() - start

    predictions = []
    for index, row in enumerate(rows):
        prediction = {
            "sample_index": index,
            "file_name": row.get("file_name"),
            "frame": row.get("frame"),
            "true_species": row.get("species"),
            "true_species_id": row.get("species_id"),
            "crop_available": False,
            "predicted_species": None,
            "ranked_species": [],
            "scores": [],
        }
        try:
            start = time.perf_counter()
            crop = _crop_from_video(row, ledger, cache_dir=cache_dir)
            timings["crop_decode_seconds_total"] += time.perf_counter() - start
            if crop is None:
                prediction["error"] = "crop unavailable"
                predictions.append(prediction)
                continue

            prediction["crop_available"] = True
            start = time.perf_counter()
            scores = _score_crop_species(model_bundle, crop, candidate_species)
            timings["inference_seconds_total"] += time.perf_counter() - start
            ranked_species = [score["species"] for score in scores]
            prediction["ranked_species"] = ranked_species
            prediction["predicted_species"] = (
                ranked_species[0] if ranked_species else None
            )
            prediction["scores"] = scores
        except Exception as exc:
            ledger.exception(
                category="eval",
                summary=f"failed to evaluate sample index {index}",
                exc=exc,
            )
            prediction["error"] = str(exc)
        predictions.append(prediction)

    metrics = compute_classification_metrics(predictions, samples_requested=sample_rows)
    performance = summarize_performance(
        samples_evaluated=metrics["samples_evaluated"],
        timings=timings,
        device_info=_device_info(model_bundle),
    )
    config = base_manifest("bioclip-eval-smoke") | {
        "dataset_id": DATASET_ID,
        "model_name": model_name,
        "load_dataset_kwargs": manifest_dataset_kwargs(kwargs),
        "sample_rows": sample_rows,
        "requested_split": split,
        "effective_split": effective_split,
        "split_fallback_reason": split_fallback_reason,
        "streaming": streaming,
        "candidate_species": candidate_species,
        "cache_dir": str(cache_dir),
    }

    config_path = write_json(output_dir / "eval_config.json", config)
    predictions_path = _write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics_path = write_json(output_dir / "metrics.json", metrics)
    performance_path = write_json(output_dir / "performance.json", performance)

    for name, path, metadata in (
        ("eval-config", config_path, {"command": "bioclip-eval-smoke"}),
        ("eval-predictions", predictions_path, {"command": "bioclip-eval-smoke"}),
        ("eval-metrics", metrics_path, {"command": "bioclip-eval-smoke"}),
        ("eval-performance", performance_path, {"command": "bioclip-eval-smoke"}),
        ("issues", ledger.path, {"command": "bioclip-eval-smoke"}),
    ):
        publish_file(
            namespace=namespace,
            key=run_scoped_key(Path(path).name),
            path=path,
            name=name,
            metadata=metadata,
        )

    safe_note(
        "BioCLIP eval smoke completed: "
        f"evaluated={metrics['samples_evaluated']}/{sample_rows}, "
        f"top1={metrics['top1_accuracy']:.3f}, "
        f"device={performance.get('device')}"
    )
    return metrics_path
