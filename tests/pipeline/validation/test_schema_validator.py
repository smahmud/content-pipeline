"""
Tests for pipeline.validation.schema_validator

Covers schema detection, transcript/enrichment/format validation,
and field-level error reporting.
"""

import pytest

from pipeline.validation.schema_validator import (
    detect_schema_type,
    validate_transcript,
    validate_enrichment,
    validate_format,
    validate_by_schema,
    SCHEMA_TRANSCRIPT,
    SCHEMA_ENRICHMENT,
    SCHEMA_FORMAT,
)


# --- Fixtures ---

@pytest.fixture
def valid_enrichment_data():
    return {
        "enrichment_version": "v1",
        "metadata": {
            "provider": "openai",
            "model": "gpt-4-turbo",
            "timestamp": "2026-02-20T10:00:00Z",
            "cost_usd": 0.05,
            "tokens_used": 1200,
            "enrichment_types": ["summary"],
        },
        "summary": {
            "short": "A short summary.",
            "medium": "A medium-length summary with more detail.",
            "long": "A long detailed summary covering all points.",
        },
    }


@pytest.fixture
def valid_transcript_data():
    return {
        "metadata": {
            "engine": "whisper",
            "engine_version": "1.0",
            "schema_version": "transcript_v1",
            "created_at": "2026-02-20T10:00:00Z",
        },
        "transcript": [
            {
                "text": "Hello world",
                "timestamp": "00:00:01.000",
            }
        ],
    }


@pytest.fixture
def valid_format_data():
    return {
        "format_version": "v1",
        "output_type": "blog",
        "timestamp": "2026-02-20T10:00:00Z",
        "source_file": "enriched.json",
        "validation": {
            "character_count": 500,
            "truncated": False,
        },
    }


# --- detect_schema_type ---

class TestDetectSchemaType:
    def test_detects_enrichment(self, valid_enrichment_data):
        assert detect_schema_type(valid_enrichment_data) == SCHEMA_ENRICHMENT

    def test_detects_format(self, valid_format_data):
        assert detect_schema_type(valid_format_data) == SCHEMA_FORMAT

    def test_detects_transcript(self, valid_transcript_data):
        assert detect_schema_type(valid_transcript_data) == SCHEMA_TRANSCRIPT

    def test_returns_none_for_unknown(self):
        assert detect_schema_type({"random": "data"}) is None

    def test_detects_transcript_by_list(self):
        data = {"transcript": [{"text": "hi", "timestamp": "00:00:00.000"}]}
        assert detect_schema_type(data) == SCHEMA_TRANSCRIPT


# --- validate_enrichment ---

class TestValidateEnrichment:
    def test_valid_enrichment(self, valid_enrichment_data):
        issues = validate_enrichment(valid_enrichment_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0

    def test_missing_version(self, valid_enrichment_data):
        del valid_enrichment_data["enrichment_version"]
        issues = validate_enrichment(valid_enrichment_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0

    def test_missing_metadata(self, valid_enrichment_data):
        del valid_enrichment_data["metadata"]
        issues = validate_enrichment(valid_enrichment_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0

    def test_missing_enrichment_content(self):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-02-20T10:00:00Z",
                "cost_usd": 0.0,
                "tokens_used": 0,
                "enrichment_types": ["summary"],
            },
        }
        issues = validate_enrichment(data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0


# --- validate_transcript ---

class TestValidateTranscript:
    def test_valid_transcript(self, valid_transcript_data):
        issues = validate_transcript(valid_transcript_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0

    def test_missing_metadata(self):
        data = {"transcript": []}
        issues = validate_transcript(data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0

    def test_invalid_segment(self, valid_transcript_data):
        valid_transcript_data["transcript"].append({"text": "bad", "timestamp": "invalid"})
        issues = validate_transcript(valid_transcript_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0


# --- validate_format ---

class TestValidateFormat:
    def test_valid_format(self, valid_format_data):
        issues = validate_format(valid_format_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0

    def test_missing_output_type(self, valid_format_data):
        del valid_format_data["output_type"]
        issues = validate_format(valid_format_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0

    def test_missing_source_file(self, valid_format_data):
        del valid_format_data["source_file"]
        issues = validate_format(valid_format_data)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) > 0


# --- validate_by_schema ---

class TestValidateBySchema:
    def test_invalid_schema_type(self):
        with pytest.raises(ValueError, match="Unknown schema type"):
            validate_by_schema({}, "invalid_type")

    def test_delegates_to_enrichment(self, valid_enrichment_data):
        issues = validate_by_schema(valid_enrichment_data, SCHEMA_ENRICHMENT)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0
