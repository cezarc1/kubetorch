import json
import sys
import types
from pathlib import Path

from qwen3_asr_orin.datasets import (
    AudioExample,
    FLEURS_DATASET_ID,
    LIBRISPEECH_DATASET_ID,
    load_manifest,
    prepare_public_manifest,
    write_manifest,
)


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
    assert load_manifest(manifest_path) == sorted(
        examples, key=lambda example: example.id
    )


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


def test_prepare_public_manifest_copies_audio_paths_without_decoding(
    monkeypatch, tmp_path: Path
):
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"audio")

    class FakeDataset(list):
        def cast_column(self, name, feature):
            assert name == "audio"
            assert feature.decode is False
            return self

    calls = []

    def fake_load_dataset(dataset_id, config, split, streaming):
        calls.append((dataset_id, config, split, streaming))
        if dataset_id == LIBRISPEECH_DATASET_ID:
            return FakeDataset(
                [{"audio": {"path": str(source_audio)}, "text": "hello"}]
            )
        return FakeDataset(
            [
                {
                    "audio": {"path": "streamed.flac", "bytes": b"audio"},
                    "transcription": "hola",
                }
            ]
        )

    class FakeAudio:
        def __init__(self, decode):
            self.decode = decode

    fake_datasets = types.SimpleNamespace(
        load_dataset=fake_load_dataset, Audio=FakeAudio
    )
    fake_soundfile = types.SimpleNamespace(
        info=lambda path: types.SimpleNamespace(duration=1.5)
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    manifest_path = prepare_public_manifest(
        output_dir=tmp_path / "out",
        librispeech_count=1,
        fleurs_count_per_language=1,
        fleurs_languages=("es_419",),
    )

    rows = load_manifest(manifest_path)
    assert calls == [
        (LIBRISPEECH_DATASET_ID, "clean", "validation", True),
        (FLEURS_DATASET_ID, "es_419", "validation", True),
    ]
    assert [row.transcript for row in rows] == ["hola", "hello"]
    for row in rows:
        copied = Path(row.audio_path)
        assert copied.exists()
        assert copied.read_bytes() == b"audio"
        assert row.duration_seconds == 1.5
    assert rows[0].audio_path.endswith(".flac")
