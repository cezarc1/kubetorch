import json

from typer.testing import CliRunner

from wetlandbirds_shakedown.cli import app


runner = CliRunner()


def test_env_probe_writes_manifest(tmp_path):
    result = runner.invoke(app, ["env-probe", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    manifest = json.loads((tmp_path / "environment.json").read_text())
    assert manifest["command"] == "env-probe"
    assert "python" in manifest
    assert "environment" in manifest
    assert (tmp_path / "issues.md").exists()


def test_cleanup_uses_prefix_deletes(monkeypatch, tmp_path):
    deleted = []

    class FakeDataStoreClient:
        def __init__(self, namespace):
            self.namespace = namespace

        def rm(self, key, recursive=False, prefix_mode=False, verbose=False):
            deleted.append(
                {
                    "namespace": self.namespace,
                    "key": key,
                    "recursive": recursive,
                    "prefix_mode": prefix_mode,
                    "verbose": verbose,
                }
            )

    monkeypatch.setattr(
        "wetlandbirds_shakedown.cleanup.DataStoreClient",
        FakeDataStoreClient,
    )

    result = runner.invoke(
        app,
        ["cleanup", "--namespace", "kubetorch", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert deleted == [
        {
            "namespace": "kubetorch",
            "key": "datasets/visual-wetlandbirds-shakedown",
            "recursive": False,
            "prefix_mode": True,
            "verbose": True,
        },
        {
            "namespace": "kubetorch",
            "key": "experiments/wetlandbirds-shakedown",
            "recursive": False,
            "prefix_mode": True,
            "verbose": True,
        },
    ]
