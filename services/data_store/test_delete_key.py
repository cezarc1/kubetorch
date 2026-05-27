import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_data_store_server():
    data_store_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(data_store_dir))
    spec = importlib.util.spec_from_file_location(
        "data_store_server_under_test", data_store_dir / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delete_key_prefix_mode_removes_nested_filesystem_prefix(monkeypatch, tmp_path):
    server = _load_data_store_server()
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("POD_NAMESPACE", "kubetorch")
    server.metadata.clear()

    artifact_dir = (
        tmp_path / "kubetorch" / "experiments" / "wetlandbirds-shakedown" / "run-1"
    )
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "metrics.json").write_text("{}\n")

    client = TestClient(server.app)
    response = client.delete(
        "/api/v1/keys/experiments/wetlandbirds-shakedown",
        params={"prefix_mode": "true"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["deleted_from_filesystem"] is True
    assert response.json()["deleted_fs_count"] == 1
    assert not (
        tmp_path / "kubetorch" / "experiments" / "wetlandbirds-shakedown"
    ).exists()
