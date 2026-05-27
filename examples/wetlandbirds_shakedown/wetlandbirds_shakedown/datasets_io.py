from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from .constants import DATASET_ID


def metadata_features() -> Any:
    from datasets import Features, Value

    return Features(
        {
            "file_name": Value("string"),
            "frame": Value("int64"),
            "x_min": Value("float64"),
            "y_min": Value("float64"),
            "x_max": Value("float64"),
            "y_max": Value("float64"),
            "behavior": Value("string"),
            "behavior_id": Value("int64"),
            "species": Value("string"),
            "species_id": Value("int64"),
            "bird_id": Value("int64"),
        }
    )


def load_dataset_kwargs(
    *,
    cache_dir: str | Path,
    dataset_id: str = DATASET_ID,
    split: str | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "cache_dir": str(cache_dir),
    }
    if split:
        kwargs["split"] = split
    if streaming:
        kwargs["streaming"] = True
        kwargs["features"] = metadata_features()
    return kwargs


def manifest_dataset_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(kwargs)
    features = serializable.get("features")
    if features is not None:
        serializable["features"] = {
            name: repr(feature) for name, feature in features.items()
        }
    return serializable


def _split_items(dataset: Any) -> list[tuple[str, Any]]:
    if isinstance(dataset, Mapping):
        return list(dataset.items())
    return [("default", dataset)]


def _sample_rows(split: Any, sample_rows: int) -> list[dict[str, Any]]:
    try:
        length = len(split)
        selected = split.select(range(min(sample_rows, length)))
        return list(selected)
    except Exception:
        return list(islice(iter(split), sample_rows))


def summarize_dataset(dataset: Any, *, sample_rows: int = 25) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset_id": DATASET_ID, "splits": {}}
    for split_name, split in _split_items(dataset):
        try:
            rows = len(split)
        except Exception:
            rows = None
        columns = list(getattr(split, "column_names", []) or [])
        samples = _sample_rows(split, sample_rows)
        species = sorted(
            {str(row["species"]) for row in samples if row.get("species") is not None}
        )
        behaviors = sorted(
            {str(row["behavior"]) for row in samples if row.get("behavior") is not None}
        )
        summary["splits"][split_name] = {
            "rows": rows,
            "columns": columns,
            "sample_rows": len(samples),
            "sample_species": species,
            "sample_behaviors": behaviors,
        }
    return summary


def disable_video_decoding(dataset: Any, *, video_cls: Any | None = None) -> Any:
    if video_cls is None:
        from datasets import Video

        video_cls = Video

    for split_name, split in _split_items(dataset):
        features = getattr(split, "features", {}) or {}
        for column, feature in features.items():
            if isinstance(feature, video_cls):
                casted = split.cast_column(column, video_cls(decode=False))
                if isinstance(dataset, Mapping):
                    dataset[split_name] = casted
                else:
                    dataset = casted
    return dataset


def choose_split(dataset: Any, preferred: str = "train") -> Any:
    if isinstance(dataset, Mapping):
        if preferred in dataset:
            return dataset[preferred]
        return next(iter(dataset.values()))
    return dataset
