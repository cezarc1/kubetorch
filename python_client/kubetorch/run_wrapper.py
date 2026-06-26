import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from kubetorch.data_store import get as kt_get, put as kt_put
from kubetorch.globals import controller_client

DEFAULT_CONTROLLER_LOG_MAX_CHARS = 200_000
READ_CHUNK_SIZE = 8192


def _controller_log_max_chars() -> int:
    raw = os.getenv("KT_CONTROLLER_LOG_MAX_CHARS", str(DEFAULT_CONTROLLER_LOG_MAX_CHARS))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_CONTROLLER_LOG_MAX_CHARS


def _append_tail(current: str, addition: str, max_chars: int) -> str:
    combined = current + addition
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def _sync_workdir(workdir_key: Optional[str], namespace: str, workdir: Path):
    if not workdir_key:
        return
    workdir.mkdir(parents=True, exist_ok=True)
    kt_get(key=workdir_key, dest=workdir, contents=True, namespace=namespace)


def _local_log_path(run_id: str) -> Path:
    safe_run_id = "".join(char if char.isalnum() or char in "-." else "-" for char in run_id)
    return Path(tempfile.gettempdir()) / f"kubetorch-{safe_run_id}-stdout.log"


def _sync_logs_to_store(logs_key: str, namespace: str, log_path: Path) -> Tuple[bool, Optional[str]]:
    if not logs_key or not log_path.exists():
        return False, None
    try:
        kt_put(key=logs_key, src=log_path, namespace=namespace, force=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _controller_log_payload(
    *,
    namespace: str,
    logs_key: str,
    tail: str,
    full_logs_synced: bool,
    sync_error: Optional[str] = None,
) -> str:
    lines = []
    if full_logs_synced:
        lines.append(f"Kubetorch stored full stdout/stderr at kt://{namespace}/{logs_key}")
    elif logs_key:
        lines.append(f"Kubetorch could not store full stdout/stderr at kt://{namespace}/{logs_key}")
        if sync_error:
            lines.append(f"Log sync error: {sync_error}")
    lines.append(f"Showing last {len(tail)} characters captured by the run wrapper:")
    lines.append("")
    lines.append(tail)
    return "\n".join(lines)


def run_wrapped_command(
    command: list[str], env: Optional[Dict[str, str]] = None, workdir: Optional[Path] = None
) -> int:
    """Run a batch command inside a run-scoped workdir and report status/logs."""
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    env = merged_env
    run_id = env["KT_RUN_ID"]
    namespace = env.get("KT_NAMESPACE", "kubetorch")
    workdir_key = env.get("KT_WORKDIR_KEY")
    logs_key = env.get("KT_LOGS_KEY", f"runs/{run_id}/logs/stdout.log")
    workdir = Path(workdir or "/workspace")
    controller = controller_client()

    max_controller_log_chars = _controller_log_max_chars()
    tail = ""
    exit_code = 1
    log_path = _local_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"")

    try:
        _sync_workdir(workdir_key, namespace, workdir)
        controller.update_run_status(run_id, "running")
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        with log_path.open("ab") as handle:
            for chunk in iter(lambda: process.stdout.read(READ_CHUNK_SIZE), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                handle.write(chunk)
                handle.flush()
                tail = _append_tail(tail, chunk.decode("utf-8", errors="replace"), max_controller_log_chars)
        exit_code = process.wait()
    except Exception as exc:
        message = f"\nKubetorch run wrapper failed: {exc}\n"
        encoded = message.encode("utf-8", errors="replace")
        with log_path.open("ab") as handle:
            handle.write(encoded)
        sys.stderr.write(message)
        tail = _append_tail(tail, message, max_controller_log_chars)
        exit_code = 1
    finally:
        status = "succeeded" if exit_code == 0 else "failed"
        full_logs_synced, sync_error = _sync_logs_to_store(logs_key, namespace, log_path)
        controller_payload = _controller_log_payload(
            namespace=namespace,
            logs_key=logs_key,
            tail=tail,
            full_logs_synced=full_logs_synced,
            sync_error=sync_error,
        )
        try:
            controller.put_run_logs(run_id, controller_payload)
        except Exception as exc:
            sys.stderr.write(f"Kubetorch could not persist controller log tail: {exc}\n")
        controller.update_run_status(run_id, status, exit_code=exit_code)
    return exit_code


def main(argv: Optional[list[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("missing command")
    raise SystemExit(run_wrapped_command(argv))


if __name__ == "__main__":
    main()
