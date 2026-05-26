import os

import pytest


@pytest.mark.level("unit")
def test_sanitize_env_redacts_secret_like_keys():
    from kubetorch.runs import sanitize_env

    sanitized = sanitize_env(
        {
            "KT_RUN_ID": "run-123",
            "TRAINING_MODE": "sft",
            "WANDB_API_KEY": "wandb-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "HF_TOKEN": "hf-secret",
            "DATABASE_PASSWORD": "db-secret",
        }
    )

    assert sanitized == {
        "KT_RUN_ID": "run-123",
        "TRAINING_MODE": "sft",
        "WANDB_API_KEY": "<redacted>",
        "AWS_SECRET_ACCESS_KEY": "<redacted>",
        "HF_TOKEN": "<redacted>",
        "DATABASE_PASSWORD": "<redacted>",
    }


@pytest.mark.level("unit")
def test_run_key_helpers_are_run_scoped():
    from kubetorch.runs import logs_key_for_run, source_key_for_run

    assert source_key_for_run("run-abc") == "runs/run-abc/workdir"
    assert logs_key_for_run("run-abc") == "runs/run-abc/logs/stdout.log"


@pytest.mark.level("unit")
def test_note_and_artifact_use_current_run_id(monkeypatch):
    from kubetorch import runs

    calls = []

    class FakeController:
        def add_run_note(self, run_id, body, author=None):
            calls.append(("note", run_id, body, author))
            return {"run_id": run_id, "body": body, "author": author}

        def add_run_artifact(self, run_id, name, uri, kind=None, metadata=None, author=None):
            calls.append(("artifact", run_id, name, uri, kind, metadata, author))
            return {"run_id": run_id, "name": name, "uri": uri, "kind": kind}

    monkeypatch.setenv("KT_RUN_ID", "run-from-env")
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())

    note = runs.note("eval regressed", author="agent")
    artifact = runs.artifact(
        "wandb-run",
        uri="wandb://entity/project/run-id",
        kind="wandb",
        metadata={"sweep": "grpo"},
        author="agent",
    )

    assert note["body"] == "eval regressed"
    assert artifact["uri"] == "wandb://entity/project/run-id"
    assert calls == [
        ("note", "run-from-env", "eval regressed", "agent"),
        (
            "artifact",
            "run-from-env",
            "wandb-run",
            "wandb://entity/project/run-id",
            "wandb",
            {"sweep": "grpo"},
            "agent",
        ),
    ]


@pytest.mark.level("unit")
def test_note_requires_run_id(monkeypatch):
    from kubetorch.runs import note

    monkeypatch.delenv("KT_RUN_ID", raising=False)

    with pytest.raises(RuntimeError, match="KT_RUN_ID"):
        note("missing run")
