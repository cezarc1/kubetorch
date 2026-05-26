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

    logs = client.put("/controller/runs/run-test-2/logs", json={"logs": "step=1 loss=0.42\n"})
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
    assert [note["body"] for note in body["notes"]] == ["loss dropped but eval regressed"]
    assert [artifact["name"] for artifact in body["artifacts"]] == ["wandb-run"]

    log_response = client.get("/controller/runs/run-test-2/logs")
    assert log_response.status_code == 200, log_response.text
    assert log_response.text == "step=1 loss=0.42\n"
