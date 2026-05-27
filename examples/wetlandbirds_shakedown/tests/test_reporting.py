import json

from wetlandbirds_shakedown.reporting import IssueLedger, kt_uri, write_json


def test_issue_ledger_writes_append_only_markdown(tmp_path):
    ledger = IssueLedger(tmp_path / "issues.md")

    ledger.add(
        category="framework",
        severity="high",
        summary="run ledger disappeared after controller restart",
        evidence="controller pod was replaced",
        workaround="export kt runs show before restore",
    )
    ledger.add(
        category="dataset",
        severity="medium",
        summary="hf cache sync is slow",
        evidence="large parquet/video cache",
        workaround="prefer PVC cache mount later",
    )

    text = ledger.path.read_text()
    assert "# WetlandBirds Shakedown Issues" in text
    assert "run ledger disappeared after controller restart" in text
    assert "hf cache sync is slow" in text
    assert text.count("| framework | high |") == 1
    assert text.count("| dataset | medium |") == 1


def test_write_json_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "manifest.json"

    write_json(target, {"dataset_id": "academic-datasets/Visual-WetlandBirds-Dataset"})

    assert json.loads(target.read_text()) == {
        "dataset_id": "academic-datasets/Visual-WetlandBirds-Dataset"
    }


def test_kt_uri_normalizes_prefix_slashes():
    assert kt_uri("kubetorch", "/experiments/run-1/result.json") == (
        "kt://kubetorch/experiments/run-1/result.json"
    )
