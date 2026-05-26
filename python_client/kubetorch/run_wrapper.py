import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from kubetorch.data_store import get as kt_get
from kubetorch.globals import controller_client


def _sync_workdir(workdir_key: Optional[str], namespace: str, workdir: Path):
    if not workdir_key:
        return
    workdir.mkdir(parents=True, exist_ok=True)
    kt_get(key=workdir_key, dest=workdir, contents=True, namespace=namespace)


def run_wrapped_command(command: list[str], env: Optional[Dict[str, str]] = None, workdir: Optional[Path] = None) -> int:
    """Run a batch command inside a run-scoped workdir and report status/logs."""
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    env = merged_env
    run_id = env["KT_RUN_ID"]
    namespace = env.get("KT_NAMESPACE", "kubetorch")
    workdir_key = env.get("KT_WORKDIR_KEY")
    workdir = Path(workdir or "/workspace")
    controller = controller_client()

    logs = ""
    exit_code = 1
    try:
        _sync_workdir(workdir_key, namespace, workdir)
        controller.update_run_status(run_id, "running")
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logs += line
        exit_code = process.wait()
    except Exception as exc:
        logs += f"\nKubetorch run wrapper failed: {exc}\n"
        sys.stderr.write(logs)
        exit_code = 1
    finally:
        status = "succeeded" if exit_code == 0 else "failed"
        try:
            controller.put_run_logs(run_id, logs)
        finally:
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
