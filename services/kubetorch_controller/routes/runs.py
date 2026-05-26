import json
import logging
from datetime import datetime, timezone
from typing import Optional

from core.database import Run, RunArtifactRef, RunNote, get_db
from core.models import (
    RunArtifactCreateRequest,
    RunArtifactResponse,
    RunCreateRequest,
    RunListResponse,
    RunLogsUpdateRequest,
    RunLogsUpdateResponse,
    RunNoteCreateRequest,
    RunNoteResponse,
    RunResponse,
    RunStatusUpdateRequest,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["runs"])

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


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
    notes = db.query(RunNote).filter(RunNote.run_id == run.run_id).order_by(RunNote.created_at.asc()).all()
    artifacts = (
        db.query(RunArtifactRef)
        .filter(RunArtifactRef.run_id == run.run_id)
        .order_by(RunArtifactRef.created_at.asc())
        .all()
    )
    return run.to_dict(notes=notes, artifacts=artifacts)


@router.post("/runs", response_model=RunResponse)
async def create_run(req: RunCreateRequest):
    """Create a durable run record before submitting a batch workload."""
    db = get_db()
    try:
        existing = db.query(Run).filter(Run.run_id == req.run_id).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Run already exists: {req.run_id}")

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
        logger.info("Created run record run_id=%s namespace=%s", run.run_id, run.namespace)
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
        return {"runs": [_run_details(db, run) for run in runs]}
    finally:
        db.close()


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    db = get_db()
    try:
        run = _get_run_or_404(db, run_id)
        return _run_details(db, run)
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
        note = RunNote(run_id=run_id, author=req.author, body=req.body, created_at=_now())
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
