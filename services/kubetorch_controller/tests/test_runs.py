import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client():
    routes_dir = Path(__file__).resolve().parents[1] / "routes"
    sys.path.insert(0, str(routes_dir))
    from runs import router

    app = FastAPI(title="runs-test")
    app.include_router(router)
    return TestClient(app)


def _create_run(client, run_id="run-test-1"):
    payload = {
        "run_id": run_id,
        "namespace": "kubetorch",
        "author": "cezar",
        "intent": "sft baseline",
        "command": ["python", "train.py", "--config", "configs/sft.yaml"],
        "source_key": f"runs/{run_id}/workdir",
        "logs_key": f"runs/{run_id}/logs/stdout.log",
        "image": "python:3.11",
        "resources": {"cpu": "1", "memory": "2Gi"},
        "env": {
            "KT_RUN_ID": run_id,
            "WANDB_API_KEY": "<redacted>",
            "TRAINING_MODE": "sft",
        },
        "job_name": f"kt-{run_id}",
    }
    response = client.post("/controller/runs", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_create_list_and_show_run():
    client = _client()

    created = _create_run(client)

    assert created["run_id"] == "run-test-1"
    assert created["status"] == "created"
    assert created["source_key"] == "runs/run-test-1/workdir"
    assert created["env"]["WANDB_API_KEY"] == "<redacted>"
    assert created["created_at"]

    listed = client.get("/controller/runs", params={"namespace": "kubetorch"})
    assert listed.status_code == 200, listed.text
    runs = listed.json()["runs"]
    assert [run["run_id"] for run in runs] == ["run-test-1"]
    assert runs[0]["intent"] == "sft baseline"
    assert runs[0]["author"] == "cezar"

    shown = client.get("/controller/runs/run-test-1")
    assert shown.status_code == 200, shown.text
    body = shown.json()
    assert body["command"] == ["python", "train.py", "--config", "configs/sft.yaml"]
    assert body["notes"] == []
    assert body["artifacts"] == []


def test_run_notes_artifacts_status_and_logs_are_attached():
    client = _client()
    _create_run(client, run_id="run-test-2")

    note = client.post(
        "/controller/runs/run-test-2/notes",
        json={"author": "agent", "body": "loss dropped but eval regressed"},
    )
    assert note.status_code == 200, note.text
    assert note.json()["body"] == "loss dropped but eval regressed"

    artifact = client.post(
        "/controller/runs/run-test-2/artifacts",
        json={
            "name": "wandb-run",
            "kind": "wandb",
            "uri": "wandb://entity/project/run-id",
            "metadata": {"sweep": "grpo"},
            "author": "agent",
        },
    )
    assert artifact.status_code == 200, artifact.text
    assert artifact.json()["uri"] == "wandb://entity/project/run-id"

    logs = client.put(
        "/controller/runs/run-test-2/logs", json={"logs": "step=1 loss=0.42\n"}
    )
    assert logs.status_code == 200, logs.text
    assert logs.json()["logs_bytes"] == len("step=1 loss=0.42\n")

    status = client.patch(
        "/controller/runs/run-test-2/status",
        json={"status": "succeeded", "exit_code": 0},
    )
    assert status.status_code == 200, status.text
    assert status.json()["status"] == "succeeded"
    assert status.json()["exit_code"] == 0
    assert status.json()["completed_at"]

    shown = client.get("/controller/runs/run-test-2")
    assert shown.status_code == 200, shown.text
    body = shown.json()
    assert [note["body"] for note in body["notes"]] == [
        "loss dropped but eval regressed"
    ]
    assert [artifact["name"] for artifact in body["artifacts"]] == ["wandb-run"]

    log_response = client.get("/controller/runs/run-test-2/logs")
    assert log_response.status_code == 200, log_response.text
    assert log_response.text == "step=1 loss=0.42\n"


def test_get_and_list_refresh_run_status_from_kubernetes(monkeypatch):
    client = _client()
    _create_run(client, run_id="run-test-3")

    import runs

    refresh_calls = []

    def fake_refresh(db, run):
        refresh_calls.append(run.run_id)
        run.status = "failed"
        run.logs = "ImagePullBackOff: failed to pull image\n"

    monkeypatch.setattr(runs, "_refresh_run_from_kubernetes", fake_refresh)

    shown = client.get("/controller/runs/run-test-3")
    assert shown.status_code == 200, shown.text
    assert shown.json()["status"] == "failed"

    listed = client.get("/controller/runs", params={"namespace": "kubetorch"})
    assert listed.status_code == 200, listed.text
    assert listed.json()["runs"][0]["status"] == "failed"
    assert refresh_calls == ["run-test-3", "run-test-3"]


def test_delete_run_removes_record_notes_artifacts_and_job(monkeypatch):
    client = _client()
    _create_run(client, run_id="run-test-4")

    client.post(
        "/controller/runs/run-test-4/notes",
        json={"author": "agent", "body": "cleanup this"},
    )
    client.post(
        "/controller/runs/run-test-4/artifacts",
        json={
            "name": "metrics",
            "kind": "json",
            "uri": "kt://kubetorch/experiments/run-test-4/metrics.json",
        },
    )

    import runs

    delete_calls = []

    def fake_delete_job(namespace, job_name):
        delete_calls.append((namespace, job_name))
        return True

    monkeypatch.setattr(runs, "_delete_run_job", fake_delete_job)

    response = client.delete("/controller/runs/run-test-4")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "run_id": "run-test-4",
        "deleted_run": True,
        "deleted_notes": 1,
        "deleted_artifacts": 1,
        "deleted_job": True,
        "job_name": "kt-run-test-4",
    }
    assert delete_calls == [("kubetorch", "kt-run-test-4")]

    missing = client.get("/controller/runs/run-test-4")
    assert missing.status_code == 404


def test_delete_missing_run_returns_404():
    client = _client()

    response = client.delete("/controller/runs/not-here")

    assert response.status_code == 404
