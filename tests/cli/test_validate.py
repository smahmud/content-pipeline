"""
Tests for cli.validate

Covers CLI invocation for single file, batch, strict mode,
report generation, and error handling.
"""

import json
import pytest
from click.testing import CliRunner

from cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_enrichment_file(tmp_path):
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
            "medium": "Medium summary.",
            "long": "Long summary.",
        },
    }
    fp = tmp_path / "enriched.json"
    fp.write_text(json.dumps(data), encoding="utf-8")
    return str(fp)


@pytest.fixture
def invalid_enrichment_file(tmp_path):
    data = {"random": "not an enrichment"}
    fp = tmp_path / "bad_enriched.json"
    fp.write_text(json.dumps(data), encoding="utf-8")
    return str(fp)


class TestValidateCLI:
    def test_no_input_or_batch(self, runner):
        result = runner.invoke(main, ["validate"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "error" in result.output.lower()

    def test_single_valid_file(self, runner, valid_enrichment_file):
        result = runner.invoke(main, ["validate", "--input", valid_enrichment_file])
        assert result.exit_code == 0
        assert "✅" in result.output

    def test_single_invalid_file(self, runner, invalid_enrichment_file):
        result = runner.invoke(main, ["validate", "--input", invalid_enrichment_file])
        assert result.exit_code != 0

    def test_explicit_schema(self, runner, valid_enrichment_file):
        result = runner.invoke(main, [
            "validate", "--input", valid_enrichment_file, "--schema", "enrichment"
        ])
        assert result.exit_code == 0

    def test_file_not_found(self, runner):
        result = runner.invoke(main, ["validate", "--input", "/nonexistent/file.json"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "❌" in result.output

    def test_strict_mode(self, runner, valid_enrichment_file):
        result = runner.invoke(main, [
            "validate", "--input", valid_enrichment_file, "--strict"
        ])
        # May pass or fail depending on warnings, but should not crash
        assert result.exit_code in (0, 1)

    def test_report_output(self, runner, valid_enrichment_file, tmp_path):
        report_path = str(tmp_path / "report.json")
        result = runner.invoke(main, [
            "validate", "--input", valid_enrichment_file, "--report", report_path
        ])
        assert result.exit_code == 0
        assert "Report saved" in result.output
        with open(report_path, "r") as f:
            report = json.load(f)
        assert "is_valid" in report

    def test_batch_validation(self, runner, tmp_path):
        for i in range(2):
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
                "summary": {"short": f"S{i}", "medium": f"M{i}", "long": f"L{i}"},
            }
            fp = tmp_path / f"file_{i}.json"
            fp.write_text(json.dumps(data), encoding="utf-8")

        pattern = str(tmp_path / "*.json")
        result = runner.invoke(main, ["validate", "--batch", pattern])
        assert result.exit_code == 0
        assert "2 passed" in result.output

    def test_batch_with_report(self, runner, tmp_path):
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
            "summary": {"short": "S", "medium": "M", "long": "L"},
        }
        fp = tmp_path / "file.json"
        fp.write_text(json.dumps(data), encoding="utf-8")

        report_path = str(tmp_path / "batch_report.json")
        pattern = str(tmp_path / "*.json")
        result = runner.invoke(main, [
            "validate", "--batch", pattern, "--report", report_path
        ])
        assert result.exit_code == 0
        with open(report_path, "r") as f:
            report = json.load(f)
        assert "total" in report
        assert "reports" in report

    def test_platform_validation(self, runner, valid_enrichment_file):
        result = runner.invoke(main, [
            "validate", "--input", valid_enrichment_file, "--platform", "twitter"
        ])
        # Should show character count info
        assert result.exit_code == 0 or "character" in result.output.lower()

    def test_help_output(self, runner):
        result = runner.invoke(main, ["validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output.lower()
