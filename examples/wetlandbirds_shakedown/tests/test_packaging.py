import tomllib
from pathlib import Path


def test_torchcodec_is_pinned_to_pytorch_25_compatible_range():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    deps = pyproject["project"]["dependencies"]

    assert "torchcodec>=0.1,<0.2" in deps


def test_dockerfile_installs_rsync_for_kubetorch_source_restore():
    dockerfile = Path("Dockerfile").read_text()

    assert "rsync" in dockerfile
