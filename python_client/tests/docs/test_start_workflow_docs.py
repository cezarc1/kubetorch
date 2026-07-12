import json
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / "python_client/kubetorch/docs/start/workflow.md"


def _record_result_source() -> str:
    guide = WORKFLOW.read_text()
    marker = "```python\n# record_result.py\n"
    assert marker in guide, "workflow page must contain the record_result.py example"
    remainder = guide.split(marker, 1)[1]
    return "# record_result.py\n" + remainder.split("\n```", 1)[0] + "\n"


def test_record_result_example_uploads_then_registers_artifact(monkeypatch, tmp_path):
    source = _record_result_source()
    compiled = compile(source, "record_result.py", "exec")
    events = []

    fake_kt = ModuleType("kubetorch")
    fake_kt.put = lambda **kwargs: events.append(("put", kwargs))
    fake_kt.artifact = lambda **kwargs: events.append(("artifact", kwargs))
    fake_kt.note = lambda body: events.append(("note", {"body": body}))

    monkeypatch.setitem(sys.modules, "kubetorch", fake_kt)
    monkeypatch.setenv("KT_RUN_ID", "demo-run")
    monkeypatch.setenv("KT_NAMESPACE", "kubetorch")
    monkeypatch.chdir(tmp_path)

    exec(compiled, {"__name__": "__main__"})

    assert [event[0] for event in events] == ["put", "artifact", "note"]
    assert events[0][1] == {
        "key": "runs/demo-run/artifacts/metrics.json",
        "src": Path("metrics.json"),
        "namespace": "kubetorch",
        "force": True,
    }
    assert events[1][1] == {
        "name": "metrics",
        "uri": "kt://kubetorch/runs/demo-run/artifacts/metrics.json",
        "kind": "kt-data-store",
        "metadata": {"content_type": "application/json"},
    }
    assert events[2][1] == {"body": "Recorded the metrics artifact."}
    assert json.loads((tmp_path / "metrics.json").read_text()) == {
        "accuracy": 0.98,
        "epochs": 3,
    }


def test_workflow_includes_launch_inspection_retrieval_and_cleanup_commands():
    guide = WORKFLOW.read_text()

    for command in (
        "kt run",
        "kt runs show RUN_ID",
        "kt runs logs RUN_ID",
        "kt runs artifact list RUN_ID",
        "kt get",
        "kt runs delete RUN_ID --dry-run",
    ):
        assert command in guide
