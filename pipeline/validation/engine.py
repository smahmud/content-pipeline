"""
Validation Engine

Central orchestrator that auto-detects schema type, delegates to
schema_validator and PlatformValidator, and returns ValidationReport.
Supports single file, batch (glob), and strict mode.
"""

import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.validation.report import ValidationIssue, ValidationReport
from pipeline.validation.schema_validator import (
    VALID_SCHEMA_TYPES,
    detect_schema_type,
    validate_by_schema,
)
from pipeline.validation.cross_reference import validate_cross_references


class ValidationEngine:
    """Central validation orchestrator for pipeline artifacts.

    Validates files against Pydantic schemas, platform limits,
    and cross-reference integrity. Supports batch processing
    and strict mode.
    """

    def __init__(self, strict: bool = False):
        """Initialize the validation engine.

        Args:
            strict: If True, warnings are treated as errors.
        """
        self.strict = strict

    def validate_file(
        self,
        file_path: str,
        schema_type: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> ValidationReport:
        """Validate a single file.

        Args:
            file_path: Path to the JSON file to validate.
            schema_type: Schema type override ('auto' or None for auto-detect).
            platform: Optional platform to validate against.

        Returns:
            ValidationReport with all issues found.
        """
        start = time.time()
        issues: List[ValidationIssue] = []

        # Load file
        data, load_issues = self._load_json(file_path)
        if load_issues:
            elapsed_ms = int((time.time() - start) * 1000)
            return ValidationReport(
                file_path=file_path,
                schema_type=schema_type or "unknown",
                is_valid=False,
                issues=load_issues,
                duration_ms=elapsed_ms,
            )

        # Detect or use provided schema type
        if not schema_type or schema_type == "auto":
            detected = detect_schema_type(data)
            if not detected:
                issues.append(ValidationIssue(
                    level="error",
                    field="root",
                    message="Cannot auto-detect schema type from file content",
                    suggestion="Use --schema to specify: transcript, enrichment, or format",
                ))
                elapsed_ms = int((time.time() - start) * 1000)
                return ValidationReport(
                    file_path=file_path,
                    schema_type="unknown",
                    is_valid=False,
                    issues=issues,
                    duration_ms=elapsed_ms,
                )
            schema_type = detected

        # Schema validation
        schema_issues = validate_by_schema(data, schema_type)
        issues.extend(schema_issues)

        # Platform validation (only for format outputs with content)
        if platform:
            platform_issues = self._validate_platform(data, platform, schema_type)
            issues.extend(platform_issues)

        # Cross-reference validation
        base_dir = str(Path(file_path).parent)
        xref_issues = validate_cross_references(data, schema_type, base_dir)
        issues.extend(xref_issues)

        # Determine validity
        has_errors = any(i.level == "error" for i in issues)
        has_warnings = any(i.level == "warning" for i in issues)
        is_valid = not has_errors and (not self.strict or not has_warnings)

        elapsed_ms = int((time.time() - start) * 1000)
        return ValidationReport(
            file_path=file_path,
            schema_type=schema_type,
            is_valid=is_valid,
            issues=issues,
            duration_ms=elapsed_ms,
        )

    def validate_batch(
        self,
        pattern: str,
        schema_type: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> List[ValidationReport]:
        """Validate multiple files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., 'outputs/*.json').
            schema_type: Schema type override.
            platform: Optional platform to validate against.

        Returns:
            List of ValidationReport, one per file.
        """
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            return [ValidationReport(
                file_path=pattern,
                schema_type=schema_type or "unknown",
                is_valid=False,
                issues=[ValidationIssue(
                    level="error",
                    field="root",
                    message=f"No files matching pattern: {pattern}",
                )],
            )]

        reports = []
        for fp in files:
            reports.append(self.validate_file(fp, schema_type, platform))
        return reports

    def _load_json(self, file_path: str):
        """Load and parse a JSON file.

        Returns:
            Tuple of (data_dict_or_None, list_of_issues).
        """
        issues: List[ValidationIssue] = []

        if not os.path.exists(file_path):
            issues.append(ValidationIssue(
                level="error",
                field="root",
                message=f"File not found: {file_path}",
            ))
            return None, issues

        if not os.path.isfile(file_path):
            issues.append(ValidationIssue(
                level="error",
                field="root",
                message=f"Path is not a file: {file_path}",
            ))
            return None, issues

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                level="error",
                field="root",
                message=f"Invalid JSON: {e}",
                suggestion="Check that the file contains valid JSON.",
            ))
            return None, issues
        except IOError as e:
            issues.append(ValidationIssue(
                level="error",
                field="root",
                message=f"Cannot read file: {e}",
            ))
            return None, issues

        return data, issues

    def _validate_platform(
        self,
        data: dict,
        platform: str,
        schema_type: str,
    ) -> List[ValidationIssue]:
        """Validate content against platform limits.

        For format outputs, validates the content field.
        For enrichment outputs, validates summary fields.
        """
        from pipeline.formatters.validator import PlatformValidator

        issues: List[ValidationIssue] = []
        pv = PlatformValidator()

        try:
            limits = pv.get_limits(platform)
        except ValueError as e:
            issues.append(ValidationIssue(
                level="error",
                field="platform",
                message=str(e),
            ))
            return issues

        # Extract content to validate based on schema type
        content = self._extract_content_for_platform(data, schema_type)
        if not content:
            issues.append(ValidationIssue(
                level="info",
                field="content",
                message="No text content found for platform validation",
            ))
            return issues

        result = pv.validate(content, platform)
        if result.exceeds_limit:
            issues.append(ValidationIssue(
                level="error",
                field="content",
                message=f"Content exceeds {platform} character limit: "
                        f"{result.character_count} > {limits.max_chars}",
                suggestion=f"Reduce content to {limits.max_chars} characters or less.",
            ))
        else:
            issues.append(ValidationIssue(
                level="info",
                field="content",
                message=f"Character count: {result.character_count}"
                        + (f" (limit: {limits.max_chars})" if limits.max_chars else " (no limit)"),
            ))

        for w in result.warnings:
            issues.append(ValidationIssue(
                level="warning",
                field="content",
                message=w,
            ))

        return issues

    def _extract_content_for_platform(
        self,
        data: dict,
        schema_type: str,
    ) -> Optional[str]:
        """Extract text content from data for platform validation."""
        # For enrichment: use summary.medium or summary.short
        if schema_type == "enrichment":
            summary = data.get("summary", {})
            if isinstance(summary, dict):
                return summary.get("medium") or summary.get("short") or summary.get("long")
            return None

        # For format: look for content field or output_content
        if schema_type == "format":
            return data.get("content") or data.get("output_content")

        return None
