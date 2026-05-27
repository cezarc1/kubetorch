from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wetlandbirds_shakedown import bioclip_eval, cli


class FakeSplit:
    features = {}
    column_names = [
        "file_name",
        "frame",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "species",
        "species_id",
    ]

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def test_bioclip_eval_smoke_writes_metrics_predictions_and_artifacts(
    monkeypatch, tmp_path
):
    rows = [
        {"file_name": "001.mp4", "frame": 0, "species": "Mallard", "species_id": 1},
        {"file_name": "002.mp4", "frame": 4, "species": "Gadwall", "species_id": 2},
    ]
    load_calls = []
    published = []
    notes = []

    def fake_load_dataset(**kwargs):
        load_calls.append(kwargs)
        return FakeSplit(rows)

    def fake_score(_model_bundle, crop, candidate_species):
        if crop == "Mallard":
            ranking = ["Mallard", "Gadwall"]
        else:
            ranking = ["Gadwall", "Mallard"]
        return [
            {"species": species, "score": 1.0 - index}
            for index, species in enumerate(ranking)
            if species in candidate_species
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(bioclip_eval, "disable_video_decoding", lambda dataset: dataset)
    monkeypatch.setattr(
        bioclip_eval,
        "_crop_from_video",
        lambda row, _ledger, cache_dir=None: row["species"],
    )
    monkeypatch.setattr(
        bioclip_eval,
        "_load_model_bundle",
        lambda model_name: {
            "model_name": model_name,
            "device_info": {
                "device": "cpu",
                "cuda_available": False,
                "cuda_device_name": None,
                "cuda_peak_memory_mb": None,
            },
        },
    )
    monkeypatch.setattr(bioclip_eval, "_score_crop_species", fake_score)
    monkeypatch.setattr(
        bioclip_eval, "publish_file", lambda **kwargs: published.append(kwargs) or True
    )
    monkeypatch.setattr(
        bioclip_eval, "safe_note", lambda body: notes.append(body) or True
    )

    result = bioclip_eval.run_bioclip_eval_smoke(
        output_dir=tmp_path,
        namespace="kubetorch",
        split="test",
        sample_rows=2,
        model_name="fake-model",
        streaming=True,
    )

    assert result == tmp_path / "metrics.json"
    assert load_calls[0]["streaming"] is True
    assert load_calls[0]["split"] == "test"

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    performance = json.loads((tmp_path / "performance.json").read_text())
    config = json.loads((tmp_path / "eval_config.json").read_text())
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]

    assert metrics["samples_requested"] == 2
    assert metrics["samples_evaluated"] == 2
    assert metrics["top1_accuracy"] == 1.0
    assert performance["device"] == "cpu"
    assert config["model_name"] == "fake-model"
    assert [prediction["predicted_species"] for prediction in predictions] == [
        "Mallard",
        "Gadwall",
    ]
    assert {item["name"] for item in published} == {
        "eval-config",
        "eval-predictions",
        "eval-metrics",
        "eval-performance",
        "issues",
    }
    assert any("BioCLIP eval smoke completed" in note for note in notes)


def test_bioclip_eval_smoke_falls_back_when_requested_hf_split_is_missing(
    monkeypatch, tmp_path
):
    attempts = []
    rows = [
        {"file_name": "001.mp4", "frame": 0, "species": "Mallard", "species_id": 1},
    ]

    def fake_load_dataset(**kwargs):
        attempts.append(kwargs["split"])
        if kwargs["split"] == "test":
            raise ValueError("Bad split: test. Available splits: ['train']")
        return FakeSplit(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(bioclip_eval, "disable_video_decoding", lambda dataset: dataset)
    monkeypatch.setattr(
        bioclip_eval,
        "_crop_from_video",
        lambda row, _ledger, cache_dir=None: row["species"],
    )
    monkeypatch.setattr(
        bioclip_eval,
        "_load_model_bundle",
        lambda _model_name: {
            "device_info": {
                "device": "cpu",
                "cuda_available": False,
                "cuda_device_name": None,
                "cuda_peak_memory_mb": None,
            },
        },
    )
    monkeypatch.setattr(
        bioclip_eval,
        "_score_crop_species",
        lambda _model_bundle, _crop, candidate_species: [
            {"species": candidate_species[0], "score": 1.0}
        ],
    )
    monkeypatch.setattr(bioclip_eval, "publish_file", lambda **_kwargs: True)
    monkeypatch.setattr(bioclip_eval, "safe_note", lambda _body: True)

    bioclip_eval.run_bioclip_eval_smoke(
        output_dir=tmp_path,
        namespace="kubetorch",
        split="test",
        sample_rows=1,
        model_name="fake-model",
        streaming=True,
    )

    config = json.loads((tmp_path / "eval_config.json").read_text())
    assert attempts == ["test", "train"]
    assert config["requested_split"] == "test"
    assert config["effective_split"] == "train"
    assert "Bad split: test" in config["split_fallback_reason"]


def test_cli_exposes_bioclip_eval_smoke(monkeypatch, tmp_path):
    runner = CliRunner()
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return Path(kwargs["output_dir"]) / "metrics.json"

    monkeypatch.setattr(cli, "run_bioclip_eval_smoke", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "bioclip-eval-smoke",
            "--output-dir",
            str(tmp_path),
            "--namespace",
            "kubetorch",
            "--split",
            "test",
            "--sample-rows",
            "3",
            "--model-name",
            "fake-model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "output_dir": tmp_path,
            "namespace": "kubetorch",
            "split": "test",
            "sample_rows": 3,
            "model_name": "fake-model",
            "streaming": True,
        }
    ]
