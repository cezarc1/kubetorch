"""Run one explicitly configured tutorial smoke test and write JSON evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs.catalog import load_catalog, Tutorial


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "python_client/kubetorch/docs/_data/catalog.yaml"


def smoke_command(tutorial: Tutorial, *, python_executable: Path) -> list[str]:
    if not tutorial.smoke_command:
        raise ValueError(f"tutorial {tutorial.id!r} has no smoke command")
    command = list(tutorial.smoke_command)
    if command[0] == "python":
        command[0] = str(python_executable)
    return command


def build_evidence(
    *,
    tutorial: Tutorial,
    command: list[str],
    exit_code: int,
    started_at: datetime,
    finished_at: datetime,
    source_commit: str,
    hardware: str,
) -> dict:
    return {
        "schema_version": 1,
        "tutorial_id": tutorial.id,
        "title": tutorial.title,
        "status": "validated" if exit_code == 0 else "failed",
        "fork_version": tutorial.validation.fork_version,
        "source_commit": source_commit,
        "hardware": hardware,
        "command": command,
        "exit_code": exit_code,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": int((finished_at - started_at).total_seconds()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorial_id")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--hardware",
        default=os.environ.get("KT_VALIDATION_HARDWARE", "unspecified"),
    )
    args = parser.parse_args()

    catalog = load_catalog(CATALOG_PATH, repo_root=REPO_ROOT)
    tutorial = next(
        (item for item in catalog.tutorials if item.id == args.tutorial_id), None
    )
    if tutorial is None:
        parser.error(f"unknown tutorial id: {args.tutorial_id}")

    try:
        command = smoke_command(tutorial, python_executable=Path(sys.executable))
    except ValueError as error:
        parser.error(str(error))

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=REPO_ROOT)
    finished_at = datetime.now(timezone.utc)
    evidence = build_evidence(
        tutorial=tutorial,
        command=command,
        exit_code=result.returncode,
        started_at=started_at,
        finished_at=finished_at,
        source_commit=source_commit,
        hardware=args.hardware,
    )
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
