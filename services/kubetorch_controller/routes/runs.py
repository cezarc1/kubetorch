import json
import logging
from datetime import datetime, timezone
from typing import Optional

from core.database import get_db, Run, RunArtifactRef, RunNote
from core.models import (
    RunArtifactCreateRequest,
    RunArtifactResponse,
    RunCreateRequest,
    RunDeleteResponse,
    RunListResponse,
    RunLogsUpdateRequest,
    RunLogsUpdateResponse,
    RunNoteCreateRequest,
    RunNoteResponse,
    RunResponse,
    RunStatusUpdateRequest,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from helpers.delete_helpers import delete_resource_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["runs"])

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
IMAGE_PULL_FAILURE_REASONS = {
    "ImagePullBackOff",
    "ErrImagePull",
    "InvalidImageName",
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
}


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.dumps(value or {})


def _get_run_or_404(db, run_id: str) -> Run:
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


def _run_details(db, run: Run):
    notes = (
        db.query(RunNote)
        .filter(RunNote.run_id == run.run_id)
        .order_by(RunNote.created_at.asc())
        .all()
    )
    artifacts = (
        db.query(RunArtifactRef)
        .filter(RunArtifactRef.run_id == run.run_id)
        .order_by(RunArtifactRef.created_at.asc())
        .all()
    )
    return run.to_dict(notes=notes, artifacts=artifacts)


def _job_status_from_kubernetes(
    namespace: str, job_name: Optional[str], run_id: str
) -> Optional[dict]:
    if not job_name:
        return None

    from core import k8s

    update = None
    try:
        if k8s.dynamic is not None:
            job_api = k8s.dynamic.resources.get(api_version="batch/v1", kind="Job")
            job = job_api.get(name=job_name, namespace=namespace)
            job_dict = job.to_dict() if hasattr(job, "to_dict") else job
            status = job_dict.get("status", {}) if isinstance(job_dict, dict) else {}
            for condition in status.get("conditions") or []:
                if (
                    condition.get("type") == "Complete"
                    and condition.get("status") == "True"
                ):
                    update = {"status": "succeeded", "exit_code": 0}
                if (
                    condition.get("type") == "Failed"
                    and condition.get("status") == "True"
                ):
                    update = {
                        "status": "failed",
                        "message": condition.get("message") or condition.get("reason"),
                    }
            if status.get("active") and update is None:
                update = {"status": "running"}
    except Exception as exc:
        logger.debug(
            "Could not refresh Job status for %s/%s: %s", namespace, job_name, exc
        )

    try:
        if k8s.core_v1 is not None:
            pods = k8s.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"kubetorch.com/run-id={run_id}",
            )
            for pod in pods.items:
                for container_status in pod.status.container_statuses or []:
                    state = container_status.state
                    waiting = getattr(state, "waiting", None)
                    running = getattr(state, "running", None)
                    terminated = getattr(state, "terminated", None)
                    if waiting and waiting.reason in IMAGE_PULL_FAILURE_REASONS:
                        message = waiting.message or waiting.reason
                        return {
                            "status": "failed",
                            "message": f"{waiting.reason}: {message}",
                        }
                    if terminated and terminated.exit_code != 0:
                        return {
                            "status": "failed",
                            "exit_code": terminated.exit_code,
                            "message": terminated.message or terminated.reason,
                        }
                    if running:
                        update = {"status": "running"}
    except Exception as exc:
        logger.debug("Could not refresh Pod status for run %s: %s", run_id, exc)

    return update


def _refresh_run_from_kubernetes(db, run: Run) -> bool:
    if run.status in TERMINAL_STATUSES:
        return False

    update = _job_status_from_kubernetes(run.namespace, run.job_name, run.run_id)
    if not update:
        return False

    now = _now()
    run.status = update["status"]
    if "exit_code" in update:
        run.exit_code = update["exit_code"]
    if update.get("message") and not run.logs:
        run.logs = update["message"] + "\n"
    if run.status == "running" and not run.started_at:
        run.started_at = now
    if run.status in TERMINAL_STATUSES and not run.completed_at:
        run.completed_at = now
    run.updated_at = now
    return True


def _delete_run_job(namespace: str, job_name: Optional[str]) -> bool:
    if not job_name:
        return False

    from core import k8s

    if k8s.dynamic is None:
        logger.debug(
            "Kubernetes dynamic client is not initialized; cannot delete Job %s/%s",
            namespace,
            job_name,
        )
        return False
    return delete_resource_sync(
        api_version="batch/v1",
        kind="Job",
        name=job_name,
        namespace=namespace,
        propagation_policy="Background",
    )


@router.post("/runs", response_model=RunResponse)
async def create_run(req: RunCreateRequest):
    """Create a durable run record before submitting a batch workload."""
    db = get_db()
    try:
        existing = db.query(Run).filter(Run.run_id == req.run_id).first()
        if existing:
            raise HTTPException(
                status_code=409, detail=f"Run already exists: {req.run_id}"
            )

        now = _now()
        run = Run(
            run_id=req.run_id,
            namespace=req.namespace,
            author=req.author,
            intent=req.intent,
            command=json.dumps(req.command),
            status="created",
            source_key=req.source_key,
            logs_key=req.logs_key,
            image=req.image,
            resources=_json(req.resources),
            env=_json(req.env),
            job_name=req.job_name,
            labels=_json(req.labels),
            annotations=_json(req.annotations),
            created_at=now,
            updated_at=now,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        logger.info(
            "Created run record run_id=%s namespace=%s", run.run_id, run.namespace
        )
        return _run_details(db, run)
    finally:
        db.close()


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    namespace: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
):
    """List run records, newest filters omitted for v0 simplicity."""
    db = get_db()
    try:
        query = db.query(Run)
        if namespace:
            query = query.filter(Run.namespace == namespace)
        if author:
            query = query.filter(Run.author == author)
        runs = query.order_by(Run.created_at.asc()).all()
        refreshed = False
        for run in runs:
            refreshed = _refresh_run_from_kubernetes(db, run) or refreshed
        if refreshed:
            db.commit()
        return {"runs": [_run_details(db, run) for run in runs]}
    finally:
        db.close()


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        if _refresh_run_from_kubernetes(db, run):
            db.commit()
            db.refresh(run)
        return _run_details(db, run)
    finally:
        db.close()


@router.delete("/runs/{run_id}", response_model=RunDeleteResponse)
async def delete_run(run_id: str, delete_job: bool = Query(True)):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        job_name = run.job_name
        deleted_job = False
        if delete_job and job_name:
            deleted_job = await run_in_threadpool(
                _delete_run_job, run.namespace, job_name
            )

        deleted_notes = (
            db.query(RunNote)
            .filter(RunNote.run_id == run_id)
            .delete(synchronize_session=False)
        )
        deleted_artifacts = (
            db.query(RunArtifactRef)
            .filter(RunArtifactRef.run_id == run_id)
            .delete(synchronize_session=False)
        )
        db.delete(run)
        db.commit()
        return {
            "run_id": run_id,
            "deleted_run": True,
            "deleted_notes": deleted_notes,
            "deleted_artifacts": deleted_artifacts,
            "deleted_job": deleted_job,
            "job_name": job_name,
        }
    finally:
        db.close()


@router.patch("/runs/{run_id}/status", response_model=RunResponse)
async def update_run_status(run_id: str, req: RunStatusUpdateRequest):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        now = _now()
        run.status = req.status
        run.exit_code = req.exit_code
        run.updated_at = now
        if req.status == "running" and not run.started_at:
            run.started_at = now
        if req.status in TERMINAL_STATUSES:
            run.completed_at = now
        db.commit()
        db.refresh(run)
        return _run_details(db, run)
    finally:
        db.close()


@router.put("/runs/{run_id}/logs", response_model=RunLogsUpdateResponse)
async def put_run_logs(run_id: str, req: RunLogsUpdateRequest):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        run.logs = req.logs
        run.updated_at = _now()
        db.commit()
        return {"run_id": run_id, "logs_bytes": len(req.logs.encode("utf-8"))}
    finally:
        db.close()


@router.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
async def get_run_logs(run_id: str):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        return PlainTextResponse(content=run.logs or "")
    finally:
        db.close()


@router.post("/runs/{run_id}/notes", response_model=RunNoteResponse)
async def add_run_note(run_id: str, req: RunNoteCreateRequest):
    db = get_db()
    try:
        _get_run_or_404(db, run_id)
        note = RunNote(
            run_id=run_id, author=req.author, body=req.body, created_at=_now()
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note.to_dict()
    finally:
        db.close()


@router.post("/runs/{run_id}/artifacts", response_model=RunArtifactResponse)
async def add_run_artifact(run_id: str, req: RunArtifactCreateRequest):
    db = get_db()
    try:
        _get_run_or_404(db, run_id)
        artifact = RunArtifactRef(
            run_id=run_id,
            name=req.name,
            kind=req.kind,
            uri=req.uri,
            artifact_metadata=_json(req.metadata),
            author=req.author,
            created_at=_now(),
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact.to_dict()
    finally:
        db.close()
