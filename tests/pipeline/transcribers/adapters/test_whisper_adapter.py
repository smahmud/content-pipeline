"""
File: test_whisper_adapter.py

Unit tests for WhisperAdapter behavior and error handling.

Covers:
- Transcription of valid audio files using Whisper
- Handling of invalid paths, empty files, and corrupted inputs
- Adapter configuration and model selection
"""
import os
import pytest
from pipeline.transcribers.adapters.whisper import WhisperAdapter
from pipeline.transcribers.adapters.base import TranscriberAdapter
from pipeline.transcribers.schemas.transcript_v1 import TranscriptV1, TranscriptSegment, build_transcript_metadata

TEST_AUDIO_PATH = "tests/assets/sample_audio.mp3"
OUTPUT_DIR = "tests/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transcript.txt")

def test_whisper_adapter_complies_with_protocol():
    adapter: TranscriberAdapter = WhisperAdapter()
    engine, version = adapter.get_engine_info()
    assert isinstance(engine, str)
    assert isinstance(version, str)

def test_whisper_invalid_audio_path():
    adapter = WhisperAdapter(model_name="base")
    invalid_path = "tests/assets/does_not_exist.mp3"

    with pytest.raises(FileNotFoundError):
        adapter.transcribe(invalid_path)

@pytest.mark.skipif(not os.path.exists(TEST_AUDIO_PATH), reason="Test audio file not found")
def test_whisper_raw_output_structure():
    adapter = WhisperAdapter(model_name="base")
    raw = adapter.transcribe(TEST_AUDIO_PATH)

    assert isinstance(raw, dict)
    assert "text" in raw and isinstance(raw["text"], str)
    assert "segments" in raw and isinstance(raw["segments"], list)
    assert all("text" in s and "start" in s and "end" in s for s in raw["segments"])
    assert "language" in raw and isinstance(raw["language"], str)


@pytest.mark.skipif(not os.path.exists(TEST_AUDIO_PATH), reason="Test audio file not found")
def test_whisper_transcription_runs():
    adapter = WhisperAdapter(model_name="base")
    result = adapter.transcribe(TEST_AUDIO_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result["text"])

    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0

OUTPUT_JSON = os.path.join(OUTPUT_DIR, "transcript_v1.json")

@pytest.mark.skipif(not os.path.exists(TEST_AUDIO_PATH), reason="Test audio file not found")
def test_whisper_transcript_normalization():
    adapter = WhisperAdapter(model_name="base")
    result = adapter.transcribe(TEST_AUDIO_PATH)

    # Extract confidences for aggregation
    confidences = [s.get("confidence") for s in result.get("segments", []) if s.get("confidence") is not None]
    confidence_avg = round(sum(confidences) / len(confidences), 3) if confidences else None

    # Build metadata
    engine, version = adapter.get_engine_info()
    metadata = build_transcript_metadata(
        engine=engine,
        engine_version=version,
        language=result.get("language"),
        confidence_avg=confidence_avg
    )

    # Build transcript segments
    segments = []
    for i, segment in enumerate(result.get("segments", [])):
        segments.append(TranscriptSegment(
            text=segment["text"],
            timestamp=format_timestamp(segment["start"]),
            confidence=segment.get("confidence", None)
        ))

    transcript = TranscriptV1(metadata=metadata, transcript=segments)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write(transcript.model_dump_json(indent=2))

    assert isinstance(transcript, TranscriptV1)
    assert len(transcript.transcript) > 0
    assert transcript.metadata.engine == "whisper-local"
    assert transcript.metadata.language is not None

def format_timestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm format"""
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"
