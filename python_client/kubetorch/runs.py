import os
from typing import Any, Dict, Optional

from kubetorch.globals import controller_client

REDACTED = "<redacted>"
SECRET_KEY_FRAGMENTS = (
    "API_KEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
    "CREDENTIAL",
)


def _is_secret_key(key: str) -> bool:
    normalized = key.upper()
    return any(fragment in normalized for fragment in SECRET_KEY_FRAGMENTS)


def sanitize_env(env: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return env values safe to persist on a run record."""
    env = dict(os.environ if env is None else env)
    return {key: REDACTED if _is_secret_key(key) else str(value) for key, value in env.items()}


def source_key_for_run(run_id: str) -> str:
    return f"runs/{run_id}/workdir"


def logs_key_for_run(run_id: str) -> str:
    return f"runs/{run_id}/logs/stdout.log"


def current_run_id(run_id: Optional[str] = None) -> str:
    resolved = run_id or os.getenv("KT_RUN_ID")
    if not resolved:
        raise RuntimeError("KT_RUN_ID is not set. Pass run_id=... or call this from inside a Kubetorch run.")
    return resolved


def note(body: str, author: Optional[str] = None, run_id: Optional[str] = None):
    """Attach an append-only note to the current run."""
    return controller_client().add_run_note(current_run_id(run_id), body=body, author=author)


def artifact(
    name: str,
    uri: str,
    kind: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """Attach a reference-only artifact pointer to the current run."""
    return controller_client().add_run_artifact(
        current_run_id(run_id),
        name=name,
        uri=uri,
        kind=kind,
        metadata=metadata,
        author=author,
    )
