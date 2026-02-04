"""
File: test_normalize.py

Unit tests for transcript normalization logic across providers.

Covers:
- Conversion of raw provider output to TranscriptV1 format
- Metadata construction and segment transformation
- Provider-specific normalization edge cases
"""
import pytest
from pipeline.transcribers.normalize import normalize_transcript_v1
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.config import WhisperLocalConfig
from pipeline.transcribers.schemas.transcript_v1 import TranscriptV1

@pytest.fixture
def provider():
    config = WhisperLocalConfig(model="base")
    return LocalWhisperProvider(config)

@pytest.fixture
def raw_basic():
    return {
        "language": "en",
        "segments": [
            {"text": "Hello world", "start": 0.0, "confidence": 0.95},
            {"text": "This is a test", "start": 2.5, "confidence": 0.90}
        ]
    }

@pytest.fixture
def raw_missing_confidence():
    return {
        "language": "en",
        "segments": [
            {"text": "No confidence here", "start": 1.0},
            {"text": "Still missing", "start": 3.0}
        ]
    }

@pytest.fixture
def raw_malformed_timestamp():
    return {
        "language": "en",
        "segments": [
            {"text": "Negative start", "start": -5.0, "confidence": 0.88}
        ]
    }

@pytest.fixture
def raw_empty_segments():
    return {
        "language": "en",
        "segments": []
    }

def test_normalize_transcript_v1_basic(provider, raw_basic):
    transcript = normalize_transcript_v1(raw_basic, provider)
    assert isinstance(transcript, TranscriptV1)
    assert transcript.metadata.language == "en"
    assert transcript.metadata.confidence_avg == 0.925
    assert transcript.transcript[0].timestamp == "00:00:00.000"
    assert transcript.transcript[1].timestamp == "00:00:02.500"

def test_normalize_transcript_v1_missing_confidence(provider, raw_missing_confidence):
    transcript = normalize_transcript_v1(raw_missing_confidence, provider)
    assert transcript.metadata.confidence_avg is None
    assert all(s.confidence is None for s in transcript.transcript)

def test_normalize_transcript_v1_malformed_timestamp(provider, raw_malformed_timestamp):
    transcript = normalize_transcript_v1(raw_malformed_timestamp, provider)
    assert transcript.transcript[0].timestamp == "00:00:00.000"  # Should clamp or format safely

def test_normalize_transcript_v1_empty_segments(provider, raw_empty_segments):
    transcript = normalize_transcript_v1(raw_empty_segments, provider)
    assert transcript.transcript == []
    assert transcript.metadata.language == "en"

def test_normalize_transcript_v1_missing_language(provider, raw_basic):
    raw = raw_basic.copy()
    raw.pop("language", None)

    transcript = normalize_transcript_v1(raw, provider)
    assert transcript.metadata.language is None

def test_normalize_transcript_v1_no_segments_key(provider):
    raw = {"language": "en"}  # no "segments" key

    transcript = normalize_transcript_v1(raw, provider)
    assert transcript.transcript == []
