from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return target


def kt_uri(namespace: str, key: str) -> str:
    return f"kt://{namespace}/{key.strip('/')}"


def _cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


class IssueLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# WetlandBirds Shakedown Issues\n\n"
                "| category | severity | summary | evidence | workaround | recorded_at |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
            )

    def add(
        self,
        *,
        category: str,
        severity: str,
        summary: str,
        evidence: str = "",
        workaround: str = "",
    ) -> None:
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self.path.open("a") as handle:
            handle.write(
                f"| {_cell(category)} | {_cell(severity)} | {_cell(summary)} | "
                f"{_cell(evidence)} | {_cell(workaround)} | {recorded_at} |\n"
            )

    def exception(self, *, category: str, summary: str, exc: BaseException) -> None:
        self.add(
            category=category,
            severity="high",
            summary=summary,
            evidence="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            workaround="inspect run logs and rerun the failed phase",
        )


def safe_note(body: str) -> bool:
    try:
        from kubetorch import runs

        runs.note(body)
        return True
    except Exception:
        return False


def safe_artifact(
    *,
    name: str,
    uri: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        from kubetorch import runs

        runs.artifact(name=name, uri=uri, kind=kind, metadata=metadata)
        return True
    except Exception:
        return False


def publish_file(
    *,
    namespace: str,
    key: str,
    path: str | Path,
    name: str,
    kind: str = "kt-data-store",
    metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        from kubetorch.data_store import DataStoreClient

        DataStoreClient(namespace=namespace).put(key=key, src=Path(path), force=True)
        return safe_artifact(
            name=name,
            uri=kt_uri(namespace, key),
            kind=kind,
            metadata=metadata,
        )
    except Exception:
        return False


def base_manifest(command: str) -> dict[str, Any]:
    return {
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kt_run_id": os.getenv("KT_RUN_ID"),
        "kt_namespace": os.getenv("KT_NAMESPACE"),
        "kt_workdir_key": os.getenv("KT_WORKDIR_KEY"),
        "kt_logs_key": os.getenv("KT_LOGS_KEY"),
    }
