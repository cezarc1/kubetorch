from __future__ import annotations

from pathlib import Path

import typer

from .bioclip_eval import run_bioclip_eval_smoke
from .bioclip_smoke import run_bioclip_smoke
from .cleanup import cleanup_datastore
from .env_probe import run_env_probe
from .ingest import run_ingest_hf
from .reporting import IssueLedger, safe_note, write_json

app = typer.Typer(add_completion=False)


def _record_failure(
    output_dir: Path, category: str, summary: str, exc: BaseException
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    IssueLedger(output_dir / "issues.md").exception(
        category=category, summary=summary, exc=exc
    )
    safe_note(f"{summary}: {exc}")


@app.command("env-probe")
def env_probe(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
) -> None:
    try:
        run_env_probe(output_dir=output_dir, namespace=namespace)
    except Exception as exc:
        _record_failure(output_dir, "env", "environment probe failed", exc)
        raise


@app.command("ingest-hf")
def ingest_hf(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
    split: str | None = typer.Option(None, "--split"),
    sync_cache_to_datastore: bool = typer.Option(True, "--sync-cache/--no-sync-cache"),
    sample_rows: int = typer.Option(25, "--sample-rows", min=1),
    streaming: bool = typer.Option(True, "--streaming/--materialize"),
) -> None:
    try:
        run_ingest_hf(
            output_dir=output_dir,
            namespace=namespace,
            split=split,
            sync_cache_to_datastore=sync_cache_to_datastore,
            sample_rows=sample_rows,
            streaming=streaming,
        )
    except Exception as exc:
        _record_failure(output_dir, "dataset", "Hugging Face ingest failed", exc)
        raise


@app.command("bioclip-smoke")
def bioclip_smoke(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
    split: str | None = typer.Option("train", "--split"),
    restore_cache_from_datastore: bool = typer.Option(
        False, "--restore-cache/--no-restore-cache"
    ),
    sample_rows: int = typer.Option(10, "--sample-rows", min=1),
    model_name: str = typer.Option(
        "hf-hub:imageomics/bioclip-2.5-vith14", "--model-name"
    ),
    streaming: bool = typer.Option(True, "--streaming/--materialize"),
) -> None:
    try:
        run_bioclip_smoke(
            output_dir=output_dir,
            namespace=namespace,
            split=split,
            restore_cache_from_datastore=restore_cache_from_datastore,
            sample_rows=sample_rows,
            model_name=model_name,
            streaming=streaming,
        )
    except Exception as exc:
        _record_failure(output_dir, "bioclip", "BioCLIP smoke failed", exc)
        raise


@app.command("bioclip-eval-smoke")
def bioclip_eval_smoke(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
    split: str | None = typer.Option("test", "--split"),
    sample_rows: int = typer.Option(50, "--sample-rows", min=1),
    model_name: str = typer.Option(
        "hf-hub:imageomics/bioclip-2.5-vith14", "--model-name"
    ),
    streaming: bool = typer.Option(True, "--streaming/--materialize"),
) -> None:
    try:
        run_bioclip_eval_smoke(
            output_dir=output_dir,
            namespace=namespace,
            split=split,
            sample_rows=sample_rows,
            model_name=model_name,
            streaming=streaming,
        )
    except Exception as exc:
        _record_failure(output_dir, "bioclip", "BioCLIP eval smoke failed", exc)
        raise


@app.command("cleanup")
def cleanup(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
) -> None:
    try:
        deleted = cleanup_datastore(namespace)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "cleanup.json",
            {"namespace": namespace, "deleted_prefixes": deleted},
        )
        safe_note(f"Cleaned shakedown data-store prefixes: {deleted}")
    except Exception as exc:
        _record_failure(output_dir, "cleanup", "cleanup failed", exc)
        raise


def main() -> None:
    app()
