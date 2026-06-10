from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    raw: dict[str, Any] | str


class TranscriptionClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds)

    def transcribe(self, audio_path: Path, response_format: str = "json") -> TranscriptionResult:
        with audio_path.open("rb") as audio_file:
            response = self.http_client.post(
                f"{self.base_url}/audio/transcriptions",
                data={"model": self.model, "response_format": response_format},
                files={"file": (audio_path.name, audio_file, _content_type(audio_path))},
            )

        if response.status_code >= 400:
            raise RuntimeError(f"transcription failed with HTTP {response.status_code}: {_response_body(response)}")

        if response_format == "text":
            return TranscriptionResult(text=response.text, raw=response.text)

        body = response.json()
        return TranscriptionResult(text=str(body.get("text", "")), raw=body)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def _response_body(response: httpx.Response) -> str:
    try:
        return json.dumps(response.json())
    except ValueError:
        return response.text
