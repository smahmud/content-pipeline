"""
Validate Subcommand Module

Validates pipeline artifacts (TranscriptV1, EnrichmentV1, FormatV1) against
their schemas and platform requirements. Supports batch validation with
glob patterns and JSON report generation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from pipeline.validation.engine import ValidationEngine
from pipeline.validation.schema_validator import VALID_SCHEMA_TYPES


logger = logging.getLogger(__name__)


@click.command(help="Validate pipeline artifacts against schemas and platform limits")
@click.option(
    "--input", "-i",
    "input_path",
    type=click.Path(),
    help="File path to validate",
)
@click.option(
    "--schema", "-s",
    "schema_type",
    type=click.Choice(["auto", "transcript", "enrichment", "format"], case_sensitive=False),
    default="auto",
    help="Schema type to validate against (default: auto-detect)",
)
@click.option(
    "--platform", "-p",
    type=str,
    help="Validate content against platform limits (e.g., twitter, linkedin)",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail on warnings in addition to errors",
)
@click.option(
    "--report", "-r",
    "report_path",
    type=click.Path(),
    help="Output JSON report to this path",
)
@click.option(
    "--batch", "-b",
    type=str,
    help="Glob pattern for batch validation (e.g., 'outputs/*.json')",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Directory for batch report output",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="WARNING",
    help="Logging level (default: WARNING)",
)
def validate(
    input_path: Optional[str],
    schema_type: str,
    platform: Optional[str],
    strict: bool,
    report_path: Optional[str],
    batch: Optional[str],
    output_dir: Optional[str],
    log_level: str,
):
    """Validate pipeline artifacts against schemas and platform limits.

    Examples:
        # Validate a single enriched file
        content-pipeline validate --input enriched.json

        # Validate with explicit schema type
        content-pipeline validate --input enriched.json --schema enrichment

        # Validate against platform limits
        content-pipeline validate --input enriched.json --platform twitter

        # Strict mode (fail on warnings)
        content-pipeline validate --input enriched.json --strict

        # Batch validation with report
        content-pipeline validate --batch "outputs/*.json" --report report.json

        # Batch validation with per-file reports
        content-pipeline validate --batch "outputs/*.json" --output-dir ./reports
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not input_path and not batch:
        click.echo("Error: --input or --batch is required", err=True)
        click.echo("Run 'content-pipeline validate --help' for usage", err=True)
        sys.exit(1)

    engine = ValidationEngine(strict=strict)

    if batch:
        _run_batch(engine, batch, schema_type, platform, report_path, output_dir)
    else:
        _run_single(engine, input_path, schema_type, platform, report_path)


def _run_single(
    engine: ValidationEngine,
    input_path: str,
    schema_type: str,
    platform: Optional[str],
    report_path: Optional[str],
):
    """Validate a single file."""
    report = engine.validate_file(input_path, schema_type, platform)

    # Display human-readable output
    click.echo(report.format_human())

    # Write JSON report if requested
    if report_path:
        _write_report(report_path, report.to_dict())
        click.echo(f"\nReport saved: {report_path}")

    if not report.is_valid:
        sys.exit(1)


def _run_batch(
    engine: ValidationEngine,
    pattern: str,
    schema_type: str,
    platform: Optional[str],
    report_path: Optional[str],
    output_dir: Optional[str],
):
    """Validate multiple files matching a glob pattern."""
    reports = engine.validate_batch(pattern, schema_type, platform)

    passed = sum(1 for r in reports if r.is_valid)
    warnings_count = sum(1 for r in reports if r.is_valid and r.warnings)
    failed = sum(1 for r in reports if not r.is_valid)

    click.echo(f"Validating {len(reports)} file(s)...")

    for report in reports:
        click.echo(report.format_human())

    # Summary
    click.echo(f"\n  ✅ {passed} passed")
    if warnings_count:
        click.echo(f"  ⚠ {warnings_count} with warnings")
    if failed:
        click.echo(f"  ❌ {failed} failed")

    # Write consolidated report
    if report_path:
        consolidated = {
            "total": len(reports),
            "passed": passed,
            "failed": failed,
            "reports": [r.to_dict() for r in reports],
        }
        _write_report(report_path, consolidated)
        click.echo(f"\nReport saved: {report_path}")

    # Write per-file reports
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for report in reports:
            fname = Path(report.file_path).stem + "_validation.json"
            fpath = str(Path(output_dir) / fname)
            _write_report(fpath, report.to_dict())
        click.echo(f"Per-file reports saved to: {output_dir}")

    if failed > 0:
        sys.exit(1)


def _write_report(path: str, data: dict):
    """Write a JSON report to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
