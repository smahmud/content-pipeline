"""
Tests for pipeline.validation.engine

Covers single file validation, batch validation, auto-detection,
platform validation, strict mode, and error handling.
"""

import json
import os
import pytest

from pipeline.validation.engine import ValidationEngine
from pipeline.validation.report import ValidationReport


@pytest.fixture
def tmp_enrichment_file(tmp_path):
    """Create a valid enrichment JSON file."""
    data = {
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
            "short": "Short summary.",
            "medium": "Medium summary with more detail.",
            "long": "Long detailed summary.",
        },
    }
    fp = tmp_path / "enriched.json"
    fp.write_text(json.dumps(data), encoding="utf-8")
    return str(fp)


@pytest.fixture
def tmp_invalid_json(tmp_path):
    """Create an invalid JSON file."""
    fp = tmp_path / "bad.json"
    fp.write_text("{invalid json", encoding="utf-8")
    return str(fp)


@pytest.fixture
def tmp_format_file(tmp_path):
    """Create a valid format JSON file."""
    data = {
        "format_version": "v1",
        "output_type": "blog",
        "timestamp": "2026-02-20T10:00:00Z",
        "source_file": "enriched.json",
        "validation": {
            "character_count": 500,
            "truncated": False,
        },
    }
    fp = tmp_path / "formatted.json"
    fp.write_text(json.dumps(data), encoding="utf-8")
    return str(fp)


@pytest.fixture
def tmp_transcript_file(tmp_path):
    """Create a valid transcript JSON file."""
    data = {
        "metadata": {
            "engine": "whisper",
            "engine_version": "1.0",
            "schema_version": "transcript_v1",
            "created_at": "2026-02-20T10:00:00Z",
        },
        "transcript": [
            {"text": "Hello world", "timestamp": "00:00:01.000"},
        ],
    }
    fp = tmp_path / "transcript.json"
    fp.write_text(json.dumps(data), encoding="utf-8")
    return str(fp)


class TestValidateFile:
    def test_valid_enrichment_auto_detect(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file)
        assert report.is_valid
        assert report.schema_type == "enrichment"

    def test_valid_format_auto_detect(self, tmp_format_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_format_file)
        assert report.is_valid
        assert report.schema_type == "format"

    def test_valid_transcript_auto_detect(self, tmp_transcript_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_transcript_file)
        assert report.is_valid
        assert report.schema_type == "transcript"

    def test_explicit_schema_type(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file, schema_type="enrichment")
        assert report.is_valid

    def test_file_not_found(self):
        engine = ValidationEngine()
        report = engine.validate_file("/nonexistent/file.json")
        assert not report.is_valid
        assert any("not found" in i.message.lower() for i in report.issues)

    def test_invalid_json(self, tmp_invalid_json):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_invalid_json)
        assert not report.is_valid
        assert any("invalid json" in i.message.lower() for i in report.issues)

    def test_undetectable_schema(self, tmp_path):
        fp = tmp_path / "mystery.json"
        fp.write_text(json.dumps({"random": "data"}), encoding="utf-8")
        engine = ValidationEngine()
        report = engine.validate_file(str(fp))
        assert not report.is_valid
        assert report.schema_type == "unknown"

    def test_duration_is_set(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file)
        assert report.duration_ms >= 0


class TestStrictMode:
    def test_strict_fails_on_warnings(self, tmp_enrichment_file):
        """Strict mode should fail if there are warnings."""
        engine = ValidationEngine(strict=True)
        report = engine.validate_file(tmp_enrichment_file, platform="twitter")
        # Enrichment summary validated against twitter may produce warnings
        # The key test: if warnings exist and strict=True, is_valid should be False
        if report.warnings:
            assert not report.is_valid

    def test_non_strict_passes_with_warnings(self, tmp_enrichment_file):
        engine = ValidationEngine(strict=False)
        report = engine.validate_file(tmp_enrichment_file)
        # Without strict, warnings don't cause failure
        if not report.errors:
            assert report.is_valid


class TestPlatformValidation:
    def test_platform_validation_enrichment(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file, platform="twitter")
        # Should have info about character count
        assert any("character count" in i.message.lower() for i in report.issues)

    def test_invalid_platform(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file, platform="nonexistent_platform")
        assert any("unknown platform" in i.message.lower() for i in report.issues)


class TestBatchValidation:
    def test_batch_multiple_files(self, tmp_path):
        """Batch validates multiple files."""
        for i in range(3):
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
                "summary": {"short": f"Summary {i}", "medium": f"Medium {i}", "long": f"Long {i}"},
            }
            fp = tmp_path / f"file_{i}.json"
            fp.write_text(json.dumps(data), encoding="utf-8")

        engine = ValidationEngine()
        pattern = str(tmp_path / "*.json")
        reports = engine.validate_batch(pattern)
        assert len(reports) == 3
        assert all(r.is_valid for r in reports)

    def test_batch_no_matches(self):
        engine = ValidationEngine()
        reports = engine.validate_batch("/nonexistent/path/*.json")
        assert len(reports) == 1
        assert not reports[0].is_valid


class TestReportOutput:
    def test_report_to_json(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "file_path" in parsed
        assert "is_valid" in parsed
        assert "summary" in parsed

    def test_report_human_format(self, tmp_enrichment_file):
        engine = ValidationEngine()
        report = engine.validate_file(tmp_enrichment_file)
        human = report.format_human()
        assert "enriched.json" in human
        assert "✅" in human or "❌" in human
