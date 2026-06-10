import json
from pathlib import Path

from qwen3_asr_orin.edgellm_results import rescore_samples_file


def test_rescore_samples_file_preserves_latency_and_keeps_first_word_after_language_prefix(tmp_path: Path):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "fleurs-en_us-0000",
                        "reference": "when you call someone",
                        "hypothesis": "language EnglishWhen you call someone.",
                        "duration_seconds": 4.0,
                        "latency_seconds": 6.0,
                        "returncode": 0,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "fleurs-fr_fr-0000",
                        "reference": "il avait raison",
                        "hypothesis": "language FrenchIl avait tort.",
                        "duration_seconds": 2.0,
                        "latency_seconds": 8.0,
                        "returncode": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = rescore_samples_file(samples_path, tmp_path / "rescored")

    rescored_rows = [
        json.loads(line)
        for line in (tmp_path / "rescored" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rescored_rows[0]["scored_hypothesis"] == "When you call someone."
    assert rescored_rows[0]["wer"] == 0.0
    assert rescored_rows[1]["scored_hypothesis"] == "Il avait tort."
    assert rescored_rows[1]["wer"] == 1 / 3
    assert summary["sample_count"] == 2
    assert summary["errors"] == 1
    assert summary["total_audio_seconds"] == 6.0
    assert summary["total_latency_seconds"] == 14.0
    assert summary["aggregate_rtf"] == 14.0 / 6.0
    assert summary["mean_wer"] == 1 / 6
