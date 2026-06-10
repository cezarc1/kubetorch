import json
from pathlib import Path

import httpx

from qwen3_asr_orin.client import TranscriptionClient


def test_transcription_client_posts_openai_audio_request(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake-wav")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "hello orin"})

    client = TranscriptionClient(
        base_url="http://sglang.local/v1",
        model="qwen3-asr",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.transcribe(audio_path, response_format="json")

    assert result.text == "hello orin"
    assert captured["url"] == "http://sglang.local/v1/audio/transcriptions"
    assert b'name="model"\r\n\r\nqwen3-asr' in captured["body"]
    assert b'name="response_format"\r\n\r\njson' in captured["body"]
    assert b'name="file"; filename="sample.wav"' in captured["body"]


def test_transcription_client_raises_with_response_body_on_http_error(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake-wav")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model loading"})

    client = TranscriptionClient(
        base_url="http://sglang.local/v1/",
        model="qwen3-asr",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        client.transcribe(audio_path)
    except RuntimeError as exc:
        assert "503" in str(exc)
        assert json.dumps({"error": "model loading"}) in str(exc)
    else:
        raise AssertionError("Expected HTTP error to raise")
