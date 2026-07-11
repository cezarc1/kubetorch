from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.docs.catalog import Tutorial, Validation
from scripts.docs.run_smoke import build_evidence, smoke_command


def tutorial(**overrides):
    values = {
        "id": "mnist",
        "title": "Train MNIST",
        "category": "Training",
        "source": "examples/tutorials/torchvision/basic_training_example.py",
        "slug": "tutorials/training/mnist",
        "hardware": ["GPU"],
        "video_id": None,
        "smoke_command": ["python", "examples/tutorials/torchvision/basic_training_example.py"],
        "validation": Validation(state="adapted", fork_version="0.5.2"),
    }
    values.update(overrides)
    return Tutorial(**values)


def test_smoke_command_uses_current_python_interpreter():
    command = smoke_command(tutorial(), python_executable=Path("/venv/bin/python"))

    assert command == [
        "/venv/bin/python",
        "examples/tutorials/torchvision/basic_training_example.py",
    ]


def test_smoke_command_rejects_reference_without_command():
    with pytest.raises(ValueError, match="has no smoke command"):
        smoke_command(tutorial(smoke_command=None), python_executable=Path("python"))


def test_build_evidence_records_result_and_hardware():
    started = datetime(2026, 7, 11, 20, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 7, 11, 20, 3, tzinfo=timezone.utc)

    evidence = build_evidence(
        tutorial=tutorial(),
        command=["python", "example.py"],
        exit_code=0,
        started_at=started,
        finished_at=finished,
        source_commit="abc123",
        hardware="2x RTX 4090",
    )

    assert evidence["status"] == "validated"
    assert evidence["tutorial_id"] == "mnist"
    assert evidence["fork_version"] == "0.5.2"
    assert evidence["hardware"] == "2x RTX 4090"
    assert evidence["duration_seconds"] == 180
