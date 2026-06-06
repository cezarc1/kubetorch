from pathlib import Path

import pytest
from typer.testing import CliRunner


runner = CliRunner()


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


@pytest.mark.level("unit")
def test_build_job_manifest_wraps_command_with_run_identity():
    from kubetorch.runs import build_job_manifest

    manifest = build_job_manifest(
        run_id="run-abc123",
        namespace="kubetorch",
        command=["python", "train.py", "--epochs", "1"],
        image="ghcr.io/run-house/server:0.5.0",
        env={"TRAINING_MODE": "sft"},
        resources={"requests": {"cpu": "1", "memory": "2Gi"}},
        image_pull_secrets=["ghcr-pull-secret"],
    )

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "kt-run-abc123"
    assert manifest["metadata"]["namespace"] == "kubetorch"
    assert manifest["metadata"]["labels"]["kubetorch.com/run-id"] == "run-abc123"

    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["restartPolicy"] == "Never"
    assert pod_spec["serviceAccountName"] == "kubetorch-service-account"
    assert pod_spec["imagePullSecrets"] == [{"name": "ghcr-pull-secret"}]

    container = pod_spec["containers"][0]
    assert container["image"] == "ghcr.io/run-house/server:0.5.0"
    assert container["command"] == ["python", "-m", "kubetorch.run_wrapper", "--"]
    assert container["args"] == ["python", "train.py", "--epochs", "1"]
    assert container["resources"] == {"requests": {"cpu": "1", "memory": "2Gi"}}
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["KT_RUN_ID"] == "run-abc123"
    assert env["KT_WORKDIR_KEY"] == "runs/run-abc123/workdir"
    assert env["KT_LOGS_KEY"] == "runs/run-abc123/logs/stdout.log"
    assert env["TRAINING_MODE"] == "sft"


@pytest.mark.level("unit")
def test_submit_batch_run_defaults_to_kubetorch_runtime_image(monkeypatch, tmp_path):
    from kubetorch import __version__, runs

    calls = []

    class FakeDataStore:
        def __init__(self, namespace):
            pass

        def put(self, key, src, contents=False, filter_options=None, force=False):
            pass

    class FakeController:
        def create_run(self, body):
            calls.append(("create-run", body))
            return {"run_id": body["run_id"], "status": "created"}

        def post(self, path, json, timeout=None):
            calls.append(("post", path, json))
            return {"status": "success"}

    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "generate_run_id", lambda name=None: "run-default-image")

    source_dir = tmp_path / "project"
    source_dir.mkdir()

    result = runs.submit_batch_run(
        command=["python", "-c", "print('ok')"], namespace="kubetorch", source_dir=source_dir
    )

    expected_image = f"ghcr.io/run-house/kubetorch:{__version__}"
    assert calls[0][1]["image"] == expected_image
    assert calls[1][2]["resource_manifest"]["spec"]["template"]["spec"]["containers"][0]["image"] == expected_image
    assert result["job_name"] == "kt-run-default-image"


@pytest.mark.level("unit")
def test_submit_batch_run_uses_cluster_config_image_pull_secrets(monkeypatch, tmp_path):
    from kubetorch import runs

    calls = []

    class FakeDataStore:
        def __init__(self, namespace):
            pass

        def put(self, key, src, contents=False, filter_options=None, force=False):
            pass

    class FakeController:
        def create_run(self, body):
            return {"run_id": body["run_id"], "status": "created"}

        def post(self, path, json, timeout=None):
            calls.append(("post", path, json))
            return {"status": "success"}

    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "generate_run_id", lambda name=None: "run-pull-secret")
    monkeypatch.setattr(runs.globals.config, "cluster_config", {"image_pull_secrets": ["ghcr-pull-secret"]})

    source_dir = tmp_path / "project"
    source_dir.mkdir()

    runs.submit_batch_run(command=["python", "train.py"], namespace="kubetorch", source_dir=source_dir)

    pod_spec = calls[0][2]["resource_manifest"]["spec"]["template"]["spec"]
    assert pod_spec["imagePullSecrets"] == [{"name": "ghcr-pull-secret"}]


@pytest.mark.level("unit")
def test_submit_batch_run_explicit_image_pull_secrets_override_cluster_defaults(monkeypatch, tmp_path):
    from kubetorch import runs

    calls = []

    class FakeDataStore:
        def __init__(self, namespace):
            pass

        def put(self, key, src, contents=False, filter_options=None, force=False):
            pass

    class FakeController:
        def create_run(self, body):
            return {"run_id": body["run_id"], "status": "created"}

        def post(self, path, json, timeout=None):
            calls.append(("post", path, json))
            return {"status": "success"}

    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "generate_run_id", lambda name=None: "run-explicit-secret")
    monkeypatch.setattr(runs.globals.config, "cluster_config", {"image_pull_secrets": ["cluster-secret"]})

    source_dir = tmp_path / "project"
    source_dir.mkdir()

    runs.submit_batch_run(
        command=["python", "train.py"],
        namespace="kubetorch",
        source_dir=source_dir,
        image_pull_secrets=["explicit-secret"],
    )

    pod_spec = calls[0][2]["resource_manifest"]["spec"]["template"]["spec"]
    assert pod_spec["imagePullSecrets"] == [{"name": "explicit-secret"}]


@pytest.mark.level("unit")
def test_submit_batch_run_marks_record_failed_when_apply_returns_error(monkeypatch, tmp_path):
    from kubetorch import runs

    calls = []

    class FakeDataStore:
        def __init__(self, namespace):
            pass

        def put(self, key, src, contents=False, filter_options=None, force=False):
            pass

    class FakeController:
        def create_run(self, body):
            calls.append(("create-run", body["run_id"]))
            return {"run_id": body["run_id"], "status": "created"}

        def post(self, path, json, timeout=None):
            calls.append(("post", path, json))
            return {"status": "error", "message": "jobs.batch is forbidden"}

        def update_run_status(self, run_id, status, exit_code=None):
            calls.append(("status", run_id, status, exit_code))
            return {"run_id": run_id, "status": status}

    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "generate_run_id", lambda name=None: "run-apply-failed")

    source_dir = tmp_path / "project"
    source_dir.mkdir()

    result = runs.submit_batch_run(command=["python", "train.py"], namespace="kubetorch", source_dir=source_dir)

    assert result["apply"]["status"] == "error"
    assert ("status", "run-apply-failed", "failed", None) in calls


@pytest.mark.level("unit")
def test_controller_chart_rbac_allows_batch_jobs():
    chart_rbac = Path(__file__).parents[2] / "charts" / "kubetorch" / "templates" / "controller" / "rbac.yaml"
    rbac_yaml = chart_rbac.read_text()

    assert 'apiGroups: ["batch"]' in rbac_yaml
    assert 'resources: ["jobs"]' in rbac_yaml


@pytest.mark.level("unit")
def test_controller_health_config_exposes_image_pull_secrets():
    chart_config = Path(__file__).parents[2] / "charts" / "kubetorch" / "templates" / "controller" / "configmap.yaml"
    nginx_config = chart_config.read_text()

    assert '"image_pull_secrets":' in nginx_config


@pytest.mark.level("unit")
def test_controller_chart_persists_sqlite_database():
    chart_root = Path(__file__).parents[2] / "charts" / "kubetorch"
    deployment = chart_root / "templates" / "controller" / "deployment.yaml"
    pvc = chart_root / "templates" / "controller" / "pvc.yaml"

    deployment_yaml = deployment.read_text()
    pvc_yaml = pvc.read_text()

    assert "mountPath: /data" in deployment_yaml
    assert "name: controller-data" in deployment_yaml
    assert "kind: PersistentVolumeClaim" in pvc_yaml
    assert "kubetorch-controller-data" in pvc_yaml


@pytest.mark.level("unit")
def test_version_mismatch_ignore_env_suppresses_warning(monkeypatch, recwarn):
    from kubetorch import __version__
    from kubetorch.provisioning.utils import check_kubetorch_versions

    class FakeResponse:
        def json(self):
            return {"version": f"{__version__}-cluster"}

    monkeypatch.setenv("KUBETORCH_IGNORE_VERSION_MISMATCH", "1")

    check_kubetorch_versions(FakeResponse())

    assert list(recwarn) == []


@pytest.mark.level("unit")
def test_submit_batch_run_creates_run_uploads_source_and_applies_job(monkeypatch, tmp_path):
    from kubetorch import runs

    calls = []

    class FakeDataStore:
        def __init__(self, namespace):
            calls.append(("data-store", namespace))

        def put(self, key, src, contents=False, filter_options=None, force=False):
            calls.append(("put", key, str(src), contents, filter_options, force))

    class FakeController:
        def create_run(self, body):
            calls.append(("create-run", body))
            return {"run_id": body["run_id"], "status": "created"}

        def post(self, path, json, timeout=None):
            calls.append(("post", path, json, timeout))
            return {
                "status": "success",
                "resource": {"metadata": {"name": json["resource_manifest"]["metadata"]["name"]}},
            }

    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)
    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "generate_run_id", lambda name=None: "run-fixed")
    monkeypatch.setattr(runs.globals.config, "username", "cezar")

    source_dir = tmp_path / "project"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('train')\n")

    result = runs.submit_batch_run(
        command=["python", "train.py"],
        namespace="kubetorch",
        source_dir=source_dir,
        image="ghcr.io/run-house/server:0.5.0",
        intent="smoke run",
        env={"WANDB_API_KEY": "secret", "TRAINING_MODE": "sft"},
    )

    assert result["run_id"] == "run-fixed"
    assert calls[0] == ("data-store", "kubetorch")
    assert calls[1][0:4] == ("put", "runs/run-fixed/workdir", str(source_dir), True)

    create_run = calls[2]
    assert create_run[0] == "create-run"
    assert create_run[1]["run_id"] == "run-fixed"
    assert create_run[1]["intent"] == "smoke run"
    assert create_run[1]["env"]["WANDB_API_KEY"] == "<redacted>"
    assert create_run[1]["env"]["TRAINING_MODE"] == "sft"

    post = calls[3]
    assert post[0:2] == ("post", "/controller/apply")
    assert post[2]["resource_type"] == "job"
    assert post[2]["resource_manifest"]["kind"] == "Job"


@pytest.mark.level("unit")
def test_cli_batch_run_help_guides_agent_operators():
    from kubetorch import cli

    run_result = runner.invoke(cli.app, ["run", "--help"], color=False)
    assert run_result.exit_code == 0, run_result.output
    assert "snapshots the source directory" in run_result.output
    assert "Kubernetes Job" in run_result.output
    assert "--intent" in run_result.output
    assert "rsync" in run_result.output

    runs_result = runner.invoke(cli.app, ["runs", "--help"], color=False)
    assert runs_result.exit_code == 0, runs_result.output
    assert "Inspect, annotate, and clean up" in runs_result.output
    assert "before launching follow-up runs" in runs_result.output

    note_result = runner.invoke(cli.app, ["runs", "note", "add", "--help"], color=False)
    assert note_result.exit_code == 0, note_result.output
    assert "findings" in note_result.output

    artifact_result = runner.invoke(cli.app, ["runs", "artifact", "add", "--help"], color=False)
    assert artifact_result.exit_code == 0, artifact_result.output
    assert "result reference" in artifact_result.output
    assert "wandb" in artifact_result.output


@pytest.mark.level("unit")
def test_batch_runs_docs_page_is_linked_and_agent_oriented():
    docs_root = Path(__file__).resolve().parents[1] / "kubetorch" / "docs"
    guide_path = docs_root / "guides" / "batch_runs.rst"
    index_path = docs_root / "index.rst"

    guide = guide_path.read_text()
    index = index_path.read_text()

    assert "guides/batch_runs" in index
    assert "Before submitting" in guide
    assert "kt runs list" in guide
    assert "kt runs delete RUN_ID --yes" in guide
    assert "not a queue or fair-share scheduler" in guide


@pytest.mark.level("unit")
def test_cli_run_submits_batch_run(monkeypatch, tmp_path):
    from kubetorch import cli

    calls = []

    def fake_submit_batch_run(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-cli", "job_name": "kt-run-cli"}

    monkeypatch.setattr(cli.runs, "submit_batch_run", fake_submit_batch_run)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--intent",
            "sft smoke",
            "--namespace",
            "kubetorch",
            "--image",
            "python:3.11",
            "--source-dir",
            str(tmp_path),
            "--env",
            "TRAINING_MODE=sft",
            "--image-pull-secret",
            "ghcr-pull-secret",
            "--",
            "python",
            "train.py",
            "--epochs",
            "1",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "run-cli" in result.output
    assert calls == [
        {
            "command": ["python", "train.py", "--epochs", "1"],
            "namespace": "kubetorch",
            "source_dir": tmp_path,
            "image": "python:3.11",
            "intent": "sft smoke",
            "env": {"TRAINING_MODE": "sft"},
            "resources": None,
            "image_pull_secrets": ["ghcr-pull-secret"],
            "name": None,
        }
    ]


@pytest.mark.level("unit")
def test_cli_runs_list_show_logs_note_and_artifact(monkeypatch):
    from kubetorch import cli

    calls = []

    class FakeController:
        def list_runs(self, namespace=None, author=None):
            calls.append(("list", namespace, author))
            return {
                "runs": [
                    {
                        "run_id": "run-1",
                        "status": "succeeded",
                        "created_at": "2026-05-26T01:00:00+00:00",
                        "author": "cezar",
                        "intent": "sft baseline",
                        "command": ["python", "train.py"],
                    }
                ]
            }

        def get_run(self, run_id):
            calls.append(("show", run_id))
            return {
                "run_id": run_id,
                "status": "succeeded",
                "source_key": f"runs/{run_id}/workdir",
                "logs_key": f"runs/{run_id}/logs/stdout.log",
                "notes": [],
                "artifacts": [],
            }

        def get_run_logs(self, run_id):
            calls.append(("logs", run_id))
            return "hello logs\n"

        def add_run_note(self, run_id, body, author=None):
            calls.append(("note", run_id, body, author))
            return {"run_id": run_id, "body": body}

        def add_run_artifact(self, run_id, name, uri, kind=None, metadata=None, author=None):
            calls.append(("artifact", run_id, name, uri, kind, metadata, author))
            return {"run_id": run_id, "name": name, "uri": uri}

    monkeypatch.setattr(cli.globals, "controller_client", lambda: FakeController())

    list_result = runner.invoke(cli.app, ["runs", "list", "--namespace", "kubetorch"], color=False)
    assert list_result.exit_code == 0, list_result.output
    assert "run-1" in list_result.output
    assert "sft baseline" in list_result.output

    show_result = runner.invoke(cli.app, ["runs", "show", "run-1"], color=False)
    assert show_result.exit_code == 0, show_result.output
    assert "runs/run-1/workdir" in show_result.output

    logs_result = runner.invoke(cli.app, ["runs", "logs", "run-1"], color=False)
    assert logs_result.exit_code == 0, logs_result.output
    assert "hello logs" in logs_result.output

    note_result = runner.invoke(cli.app, ["runs", "note", "add", "run-1", "looks good"], color=False)
    assert note_result.exit_code == 0, note_result.output

    artifact_result = runner.invoke(
        cli.app,
        ["runs", "artifact", "add", "run-1", "--name", "wandb", "--uri", "wandb://entity/project/run"],
        color=False,
    )
    assert artifact_result.exit_code == 0, artifact_result.output

    assert ("list", "kubetorch", None) in calls
    assert ("show", "run-1") in calls
    assert ("logs", "run-1") in calls
    assert ("note", "run-1", "looks good", None) in calls
    assert ("artifact", "run-1", "wandb", "wandb://entity/project/run", None, None, None) in calls


@pytest.mark.level("unit")
def test_delete_batch_run_plans_and_deletes_run_data(monkeypatch):
    from kubetorch import runs

    calls = []

    run_record = {
        "run_id": "run-1",
        "namespace": "kubetorch",
        "source_key": "runs/run-1/workdir",
        "logs_key": "runs/run-1/logs/stdout.log",
        "artifacts": [
            {"name": "metrics", "uri": "kt://kubetorch/experiments/wetlandbirds-shakedown/run-1/metrics.json"},
            {"name": "wandb", "uri": "wandb://entity/project/run-1"},
        ],
    }

    class FakeController:
        def get_run(self, run_id):
            calls.append(("get-run", run_id))
            return run_record

        def delete_run(self, run_id, delete_job=True):
            calls.append(("delete-run", run_id, delete_job))
            return {"run_id": run_id, "deleted_run": True, "deleted_job": delete_job}

    class FakeDataStore:
        def __init__(self, namespace):
            calls.append(("data-store", namespace))

        def rm(self, key, recursive=False, prefix_mode=False, verbose=False):
            calls.append(("rm", key, recursive, prefix_mode, verbose))

    monkeypatch.setattr(runs, "controller_client", lambda: FakeController())
    monkeypatch.setattr(runs, "DataStoreClient", FakeDataStore)

    dry_run = runs.delete_batch_run("run-1", dry_run=True)

    assert dry_run["data_keys"] == ["runs/run-1", "experiments/wetlandbirds-shakedown/run-1"]
    assert ("delete-run", "run-1", True) not in calls

    result = runs.delete_batch_run("run-1", yes=True)

    assert result["data_keys"] == ["runs/run-1", "experiments/wetlandbirds-shakedown/run-1"]
    assert ("rm", "runs/run-1", True, False, False) in calls
    assert ("rm", "experiments/wetlandbirds-shakedown/run-1", True, False, False) in calls
    assert ("delete-run", "run-1", True) in calls


@pytest.mark.level("unit")
def test_cli_runs_delete_uses_delete_batch_run(monkeypatch):
    from kubetorch import cli

    calls = []

    def fake_delete_batch_run(run_id, delete_data=True, delete_job=True, dry_run=False, yes=False):
        calls.append((run_id, delete_data, delete_job, dry_run, yes))
        return {
            "run_id": run_id,
            "data_keys": ["runs/run-1"],
            "controller": {"deleted_run": not dry_run, "deleted_job": delete_job},
        }

    monkeypatch.setattr(cli.runs, "delete_batch_run", fake_delete_batch_run)

    dry_result = runner.invoke(cli.app, ["runs", "delete", "run-1", "--dry-run"], color=False)
    assert dry_result.exit_code == 0, dry_result.output
    assert "runs/run-1" in dry_result.output

    delete_result = runner.invoke(cli.app, ["runs", "delete", "run-1", "--yes"], color=False)
    assert delete_result.exit_code == 0, delete_result.output
    assert "Deleted run run-1" in delete_result.output

    assert calls == [
        ("run-1", True, True, True, False),
        ("run-1", True, True, False, True),
    ]


@pytest.mark.level("unit")
def test_run_wrapper_syncs_workdir_persists_logs_and_status(monkeypatch, tmp_path):
    from kubetorch import run_wrapper

    calls = []

    class FakeController:
        def update_run_status(self, run_id, status, exit_code=None):
            calls.append(("status", run_id, status, exit_code))
            return {"run_id": run_id, "status": status}

        def put_run_logs(self, run_id, logs):
            calls.append(("logs", run_id, logs))
            return {"run_id": run_id, "logs_bytes": len(logs)}

    def fake_get(key, dest, contents, namespace):
        calls.append(("get", key, str(dest), contents, namespace))
        (tmp_path / "train.py").write_text("print('from synced source')\n")

    monkeypatch.setattr(run_wrapper, "controller_client", lambda: FakeController())
    monkeypatch.setattr(run_wrapper, "kt_get", fake_get)

    exit_code = run_wrapper.run_wrapped_command(
        ["python", "train.py"],
        env={
            "KT_RUN_ID": "run-wrapper",
            "KT_NAMESPACE": "kubetorch",
            "KT_WORKDIR_KEY": "runs/run-wrapper/workdir",
        },
        workdir=tmp_path,
    )

    assert exit_code == 0
    assert calls[0] == ("get", "runs/run-wrapper/workdir", str(tmp_path), True, "kubetorch")
    assert calls[1] == ("status", "run-wrapper", "running", None)
    assert calls[2][0:2] == ("logs", "run-wrapper")
    assert "from synced source" in calls[2][2]
    assert calls[3] == ("status", "run-wrapper", "succeeded", 0)
