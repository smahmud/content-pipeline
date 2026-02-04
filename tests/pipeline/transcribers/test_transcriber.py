"""
File: test_transcriber.py

Adapter-level edge case tests using generic audio inputs.

Covers:
- LocalWhisperProvider behavior on empty audio files
- LocalWhisperProvider behavior on corrupted or non-audio input
"""
from unittest.mock import patch
import pytest
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.config import WhisperLocalConfig
from pipeline.transcribers.normalize import normalize_transcript_v1

def test_transcribe_empty_audio_file(tmp_path):
    empty_audio = tmp_path / "empty.mp3"
    empty_audio.write_bytes(b"")  # Create a zero-byte file

    config = WhisperLocalConfig(model="base")
    provider = LocalWhisperProvider(config)
    with pytest.raises(Exception):  # Or a specific error type
        provider.transcribe(str(empty_audio))

def test_transcribe_invalid_audio_file(tmp_path):
    bad_audio = tmp_path / "invalid.mp3"
    bad_audio.write_text("not real audio")

    config = WhisperLocalConfig(model="base")
    provider = LocalWhisperProvider(config)
    with pytest.raises(Exception):
        provider.transcribe(str(bad_audio))

@patch("pipeline.transcription.providers.local_whisper.LocalWhisperProvider.transcribe")
def test_transcribe_mocked_response(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"text": "Hello world", "start": 0.0, "end": 1.0}
        ],
        "language": "en"
    }

    config = WhisperLocalConfig(model="base")
    provider = LocalWhisperProvider(config)
    raw = provider.transcribe("fake.mp3", language="en")

    assert "segments" in raw
    assert raw["segments"][0]["text"] == "Hello world"

    normalized = normalize_transcript_v1(raw, provider)
    assert normalized.transcript[0].text == "Hello world"
    assert normalized.metadata.language == "en"


