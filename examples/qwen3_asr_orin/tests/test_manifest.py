import json
from pathlib import Path

from qwen3_asr_orin.datasets import AudioExample, FLEURS_DATASET_ID, LIBRISPEECH_DATASET_ID, load_manifest, write_manifest


def test_dataset_ids_are_namespaced_for_current_hugging_face_hub():
    assert LIBRISPEECH_DATASET_ID == "openslr/librispeech_asr"
    assert FLEURS_DATASET_ID == "google/fleurs"


def test_write_manifest_sorts_examples_and_preserves_required_fields(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    examples = [
        AudioExample(
            id="fleurs-0002",
            dataset="fleurs",
            language="fr",
            split="validation",
            audio_path="/data/fleurs-0002.wav",
            transcript="bonjour orin",
            duration_seconds=2.5,
        ),
        AudioExample(
            id="librispeech-0001",
            dataset="librispeech",
            language="en",
            split="dev-clean",
            audio_path="/data/librispeech-0001.flac",
            transcript="hello orin",
            duration_seconds=1.25,
        ),
    ]

    write_manifest(examples, manifest_path)

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["fleurs-0002", "librispeech-0001"]
    assert rows[0] == {
        "id": "fleurs-0002",
        "dataset": "fleurs",
        "language": "fr",
        "split": "validation",
        "audio_path": "/data/fleurs-0002.wav",
        "transcript": "bonjour orin",
        "duration_seconds": 2.5,
    }
    assert load_manifest(manifest_path) == sorted(examples, key=lambda example: example.id)


def test_audio_example_rejects_non_positive_duration():
    try:
        AudioExample(
            id="bad",
            dataset="librispeech",
            language="en",
            split="dev-clean",
            audio_path="/tmp/bad.wav",
            transcript="bad sample",
            duration_seconds=0,
        )
    except ValueError as exc:
        assert "duration_seconds" in str(exc)
    else:
        raise AssertionError("AudioExample accepted a zero-duration sample")
