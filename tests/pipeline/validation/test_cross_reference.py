"""
Tests for pipeline.validation.cross_reference
"""

import pytest

from pipeline.validation.cross_reference import validate_cross_references


class TestCrossReferenceValidator:
    def test_format_missing_source_file(self, tmp_path):
        data = {"source_file": "nonexistent.json"}
        issues = validate_cross_references(data, "format", str(tmp_path))
        warnings = [i for i in issues if i.level == "warning"]
        assert len(warnings) == 1
        assert "not found" in warnings[0].message.lower()

    def test_format_existing_source_file(self, tmp_path):
        (tmp_path / "enriched.json").write_text("{}")
        data = {"source_file": "enriched.json"}
        issues = validate_cross_references(data, "format", str(tmp_path))
        warnings = [i for i in issues if i.level == "warning"]
        assert len(warnings) == 0

    def test_enrichment_mismatched_types(self):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "enrichment_types": ["summary", "tags"],
            },
            "summary": {"short": "test"},
            # tags is missing
        }
        issues = validate_cross_references(data, "enrichment")
        warnings = [i for i in issues if i.level == "warning"]
        assert any("tags" in w.message for w in warnings)

    def test_enrichment_consistent_types(self):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "enrichment_types": ["summary"],
            },
            "summary": {"short": "test"},
        }
        issues = validate_cross_references(data, "enrichment")
        warnings = [i for i in issues if i.level == "warning"]
        assert len(warnings) == 0

    def test_transcript_no_cross_refs(self):
        """Transcript type has no cross-reference checks."""
        issues = validate_cross_references({}, "transcript")
        assert len(issues) == 0
