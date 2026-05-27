from __future__ import annotations

from pathlib import Path

from .constants import DATASET_ID, HF_CACHE_KEY, run_scoped_key
from .datasets_io import (
    disable_video_decoding,
    load_dataset_kwargs,
    manifest_dataset_kwargs,
    summarize_dataset,
)
from .reporting import base_manifest, IssueLedger, publish_file, safe_note, write_json


def run_ingest_hf(
    *,
    output_dir: Path,
    namespace: str,
    split: str | None,
    sync_cache_to_datastore: bool,
    sample_rows: int,
    streaming: bool = True,
) -> Path:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "hf-cache"
    ledger = IssueLedger(output_dir / "issues.md")
    kwargs = load_dataset_kwargs(cache_dir=cache_dir, split=split, streaming=streaming)

    safe_note(f"Starting Hugging Face dataset ingest for {DATASET_ID}")
    dataset = load_dataset(**kwargs)
    dataset = disable_video_decoding(dataset)
    summary = summarize_dataset(dataset, sample_rows=sample_rows)

    manifest = base_manifest("ingest-hf") | {
        "dataset_id": DATASET_ID,
        "load_dataset_kwargs": manifest_dataset_kwargs(kwargs),
        "cache_dir": str(cache_dir),
        "sync_cache_to_datastore": sync_cache_to_datastore,
        "streaming": streaming,
        "cache_sync_skipped": sync_cache_to_datastore and streaming,
        "cache_sync_skip_reason": (
            "Hugging Face streaming datasets do not create a complete materialized cache"
            if sync_cache_to_datastore and streaming
            else None
        ),
        "summary": summary,
    }
    target = write_json(output_dir / "download_manifest.json", manifest)
    publish_file(
        namespace=namespace,
        key=run_scoped_key("download_manifest.json"),
        path=target,
        name="download-manifest",
        metadata={"dataset_id": DATASET_ID},
    )

    if sync_cache_to_datastore and streaming:
        ledger.add(
            category="dataset",
            severity="medium",
            summary="skipped Kubetorch data-store upload of the Hugging Face cache",
            evidence=(
                "ingest-hf ran with streaming=True to avoid materializing video Arrow files "
                "through the torchcodec decoder stack"
            ),
            workaround="rerun with --materialize after selecting a compatible torch/torchcodec/datasets stack",
        )
        safe_note(
            "Skipped Hugging Face cache upload because streaming mode has no full local cache"
        )

    if sync_cache_to_datastore and not streaming:
        try:
            from kubetorch.data_store import DataStoreClient

            DataStoreClient(namespace=namespace).put(
                key=HF_CACHE_KEY,
                src=cache_dir,
                contents=True,
                force=True,
            )
            safe_note(f"Uploaded Hugging Face cache to kt://{namespace}/{HF_CACHE_KEY}")
        except Exception as exc:
            ledger.exception(
                category="data-store",
                summary="failed to upload Hugging Face cache to Kubetorch data store",
                exc=exc,
            )
            raise

    publish_file(
        namespace=namespace,
        key=run_scoped_key("issues.md"),
        path=ledger.path,
        name="issues",
        metadata={"command": "ingest-hf"},
    )
    safe_note("Hugging Face dataset ingest completed")
    return target
