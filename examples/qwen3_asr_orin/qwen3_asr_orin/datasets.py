from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

LIBRISPEECH_DATASET_ID = "openslr/librispeech_asr"
FLEURS_DATASET_ID = "google/fleurs"


@dataclass(frozen=True)
class AudioExample:
    id: str
    dataset: str
    language: str
    split: str
    audio_path: str
    transcript: str
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not self.id:
            raise ValueError("id is required")
        if not self.audio_path:
            raise ValueError("audio_path is required")


def write_manifest(examples: Iterable[AudioExample], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(examples, key=lambda example: example.id)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for example in ordered:
            manifest_file.write(json.dumps(asdict(example), sort_keys=True) + "\n")


def load_manifest(manifest_path: Path) -> list[AudioExample]:
    examples = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                examples.append(AudioExample(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid manifest row {line_number} in {manifest_path}: {exc}") from exc
    return examples


def prepare_public_manifest(
    output_dir: Path,
    librispeech_count: int = 8,
    fleurs_count_per_language: int = 2,
    fleurs_languages: tuple[str, ...] = ("en_us", "es_419", "fr_fr", "de_de"),
) -> Path:
    """Download a small LibriSpeech + FLEURS subset and write a local manifest.

    This function imports heavy dataset/audio dependencies lazily so unit tests
    and summary analysis stay fast.
    """
    from datasets import load_dataset
    import soundfile as sf

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    examples: list[AudioExample] = []

    librispeech = load_dataset(LIBRISPEECH_DATASET_ID, "clean", split=f"validation[:{librispeech_count}]")
    for index, row in enumerate(librispeech):
        audio = row["audio"]
        audio_path = audio_dir / f"librispeech-dev-clean-{index:04d}.wav"
        sf.write(audio_path, audio["array"], audio["sampling_rate"])
        examples.append(
            AudioExample(
                id=f"librispeech-dev-clean-{index:04d}",
                dataset="librispeech",
                language="en",
                split="dev-clean",
                audio_path=str(audio_path),
                transcript=row["text"],
                duration_seconds=len(audio["array"]) / float(audio["sampling_rate"]),
            )
        )

    for language in fleurs_languages:
        fleurs = load_dataset(FLEURS_DATASET_ID, language, split=f"validation[:{fleurs_count_per_language}]")
        for index, row in enumerate(fleurs):
            audio = row["audio"]
            audio_path = audio_dir / f"fleurs-{language}-{index:04d}.wav"
            sf.write(audio_path, audio["array"], audio["sampling_rate"])
            examples.append(
                AudioExample(
                    id=f"fleurs-{language}-{index:04d}",
                    dataset="fleurs",
                    language=language,
                    split="validation",
                    audio_path=str(audio_path),
                    transcript=row["transcription"],
                    duration_seconds=len(audio["array"]) / float(audio["sampling_rate"]),
                )
            )

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(examples, manifest_path)
    return manifest_path
