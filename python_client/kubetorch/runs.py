import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from kubetorch import globals
from kubetorch.data_store import DataStoreClient
from kubetorch.globals import controller_client
from kubetorch.provisioning import constants as provisioning_constants

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


def run_data_prefix_for_run(run_id: str) -> str:
    return f"runs/{run_id}"


def generate_run_id(name: Optional[str] = None) -> str:
    prefix = re.sub(r"[^a-z0-9-]+", "-", (name or "run").lower()).strip("-")
    prefix = prefix or "run"
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _job_name_for_run(run_id: str) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    safe = safe or "run"
    return f"kt-{safe}"[:63].rstrip("-")


def _normalize_image_pull_secrets(image_pull_secrets: Optional[list[Any]]) -> Optional[list[str]]:
    if not image_pull_secrets:
        return None

    normalized = []
    for secret in image_pull_secrets:
        name = secret.get("name") if isinstance(secret, dict) else str(secret)
        if name and name not in normalized:
            normalized.append(name)
    return normalized or None


def resolve_image_pull_secrets(image_pull_secrets: Optional[list[str]] = None) -> Optional[list[str]]:
    explicit = _normalize_image_pull_secrets(image_pull_secrets)
    if explicit:
        return explicit

    cluster_config = globals.config.cluster_config or {}
    return _normalize_image_pull_secrets(
        cluster_config.get("image_pull_secrets") or cluster_config.get("imagePullSecrets")
    )


def _mark_run_submission_failed(controller, run_id: str) -> None:
    if not hasattr(controller, "update_run_status"):
        return
    try:
        controller.update_run_status(run_id, "failed")
    except Exception:
        pass


def build_job_manifest(
    run_id: str,
    namespace: str,
    command: list[str],
    image: str,
    env: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    image_pull_secrets: Optional[list[str]] = None,
    service_account_name: str = provisioning_constants.DEFAULT_SERVICE_ACCOUNT_NAME,
) -> Dict[str, Any]:
    """Build the Kubernetes Job that executes a run-scoped command."""
    job_name = _job_name_for_run(run_id)
    run_labels = {
        "app": job_name,
        "kubetorch.com/run-id": run_id,
        provisioning_constants.KT_USERNAME_LABEL: globals.config.username or "",
    }
    if labels:
        run_labels.update(labels)

    run_env = {
        "KT_RUN_ID": run_id,
        "KT_NAMESPACE": namespace,
        "KT_WORKDIR_KEY": source_key_for_run(run_id),
        "KT_LOGS_KEY": logs_key_for_run(run_id),
    }
    if env:
        run_env.update({key: str(value) for key, value in env.items()})

    container = {
        "name": "run",
        "image": image,
        "command": ["python", "-m", "kubetorch.run_wrapper", "--"],
        "args": command,
        "env": [{"name": key, "value": value} for key, value in run_env.items()],
    }
    if resources:
        container["resources"] = resources

    metadata = {"name": job_name, "namespace": namespace, "labels": run_labels}
    pod_metadata = {"labels": run_labels}
    if annotations:
        metadata["annotations"] = annotations
        pod_metadata["annotations"] = annotations

    pod_spec = {
        "restartPolicy": "Never",
        "serviceAccountName": service_account_name,
        "containers": [container],
    }
    if image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": secret_name} for secret_name in image_pull_secrets]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": pod_metadata,
                "spec": pod_spec,
            },
        },
    }


def submit_batch_run(
    command: list[str],
    namespace: Optional[str] = None,
    source_dir: Optional[Path] = None,
    image: Optional[str] = None,
    intent: Optional[str] = None,
    env: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    image_pull_secrets: Optional[list[str]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a run record, upload source, and submit a Kubernetes Job."""
    if not command:
        raise ValueError("command is required")

    namespace = namespace or globals.config.namespace
    source_dir = Path(source_dir or Path.cwd()).resolve()
    image = image or provisioning_constants.KUBETORCH_IMAGE
    run_id = generate_run_id(name)
    source_key = source_key_for_run(run_id)
    logs_key = logs_key_for_run(run_id)
    resolved_image_pull_secrets = resolve_image_pull_secrets(image_pull_secrets)

    data_store = DataStoreClient(namespace=namespace)
    data_store.put(
        key=source_key,
        src=source_dir,
        contents=True,
        filter_options="--exclude='.git' --exclude='__pycache__'",
        force=True,
    )

    manifest = build_job_manifest(
        run_id=run_id,
        namespace=namespace,
        command=command,
        image=image,
        env=env,
        resources=resources,
        labels=labels,
        annotations=annotations,
        image_pull_secrets=resolved_image_pull_secrets,
    )
    job_name = manifest["metadata"]["name"]
    run_payload = {
        "run_id": run_id,
        "namespace": namespace,
        "author": globals.config.username,
        "intent": intent,
        "command": command,
        "source_key": source_key,
        "logs_key": logs_key,
        "image": image,
        "resources": resources or {},
        "env": sanitize_env(env or {}),
        "job_name": job_name,
        "labels": labels or {},
        "annotations": annotations or {},
    }

    controller = controller_client()
    run_record = controller.create_run(run_payload)
    try:
        apply_response = controller.post(
            "/controller/apply",
            json={
                "service_name": job_name,
                "namespace": namespace,
                "resource_type": "job",
                "resource_manifest": manifest,
            },
            timeout=None,
        )
    except Exception:
        _mark_run_submission_failed(controller, run_id)
        raise

    if isinstance(apply_response, dict) and apply_response.get("status") not in (None, "success"):
        _mark_run_submission_failed(controller, run_id)
    return {"run_id": run_id, "run": run_record, "apply": apply_response, "job_name": job_name}


def _artifact_cleanup_key(artifact: Dict[str, Any], run_id: str, namespace: Optional[str]) -> Optional[str]:
    uri = artifact.get("uri")
    if not uri:
        return None

    parsed = urlparse(uri)
    if parsed.scheme != "kt":
        return None
    if namespace and parsed.netloc and parsed.netloc != namespace:
        return None

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if run_id not in parts:
        return None
    return "/".join(parts[: parts.index(run_id) + 1])


def _run_data_keys(run_record: Dict[str, Any]) -> list[str]:
    run_id = run_record["run_id"]
    namespace = run_record.get("namespace")
    keys = [run_data_prefix_for_run(run_id)]
    for artifact_ref in run_record.get("artifacts") or []:
        cleanup_key = _artifact_cleanup_key(artifact_ref, run_id=run_id, namespace=namespace)
        if cleanup_key and cleanup_key not in keys:
            keys.append(cleanup_key)
    return keys


def delete_batch_run(
    run_id: str,
    delete_data: bool = True,
    delete_job: bool = True,
    dry_run: bool = False,
    yes: bool = False,
) -> Dict[str, Any]:
    """Delete a batch run record and, by default, its Job plus run-scoped data."""
    controller = controller_client()
    run_record = controller.get_run(run_id)
    data_keys = _run_data_keys(run_record)
    result = {
        "run_id": run_id,
        "run": run_record,
        "data_keys": data_keys if delete_data else [],
        "delete_job": delete_job,
        "controller": None,
    }
    if dry_run:
        return result

    if delete_data:
        data_store = DataStoreClient(namespace=run_record.get("namespace"))
        for key in data_keys:
            data_store.rm(key, recursive=True)

    result["controller"] = controller.delete_run(run_id, delete_job=delete_job)
    return result


def current_run_id(run_id: Optional[str] = None) -> str:
    resolved = run_id or os.getenv("KT_RUN_ID")
    if not resolved:
        raise RuntimeError("KT_RUN_ID is not set. Pass run_id=... or call this from inside a Kubetorch run.")
    return resolved


def note(body: str, author: Optional[str] = None, run_id: Optional[str] = None):
    """Attach an append-only note to the current run."""
    resolved_run_id = current_run_id(run_id)
    return controller_client().add_run_note(resolved_run_id, body=body, author=author)


def artifact(
    name: str,
    uri: str,
    kind: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """Attach a reference-only artifact pointer to the current run."""
    resolved_run_id = current_run_id(run_id)
    return controller_client().add_run_artifact(
        resolved_run_id,
        name=name,
        uri=uri,
        kind=kind,
        metadata=metadata,
        author=author,
    )
