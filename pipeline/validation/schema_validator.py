"""
Schema Validator

Wraps Pydantic schemas (TranscriptV1, EnrichmentV1, FormatV1) for validation
with field-level error details. Returns ValidationIssue lists instead of
raising exceptions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from pipeline.validation.report import ValidationIssue


# Schema type constants
SCHEMA_TRANSCRIPT = "transcript"
SCHEMA_ENRICHMENT = "enrichment"
SCHEMA_FORMAT = "format"

VALID_SCHEMA_TYPES = [SCHEMA_TRANSCRIPT, SCHEMA_ENRICHMENT, SCHEMA_FORMAT]


def _pydantic_errors_to_issues(
    errors: list,
    schema_type: str,
) -> List[ValidationIssue]:
    """Convert Pydantic validation errors to ValidationIssue list."""
    issues = []
    for err in errors:
        field_path = ".".join(str(loc) for loc in err["loc"]) if err["loc"] else "root"
        issues.append(ValidationIssue(
            level="error",
            field=field_path,
            message=f"{err['msg']} (type={err['type']})",
            suggestion=f"Check the '{field_path}' field in your {schema_type} file.",
        ))
    return issues


def validate_transcript(data: Dict[str, Any]) -> List[ValidationIssue]:
    """Validate data against TranscriptV1 schema.

    Args:
        data: Parsed JSON data to validate.

    Returns:
        List of ValidationIssue (empty if valid).
    """
    from pipeline.transcribers.schemas.transcript_v1 import TranscriptV1

    issues: List[ValidationIssue] = []
    try:
        TranscriptV1(**data)
    except ValidationError as e:
        issues.extend(_pydantic_errors_to_issues(e.errors(), SCHEMA_TRANSCRIPT))
    except Exception as e:
        issues.append(ValidationIssue(
            level="error",
            field="root",
            message=f"Unexpected validation error: {e}",
        ))
    return issues


def validate_enrichment(data: Dict[str, Any]) -> List[ValidationIssue]:
    """Validate data against EnrichmentV1 schema.

    Uses the existing InputValidator for structural checks plus
    Pydantic schema validation for field-level detail.

    Args:
        data: Parsed JSON data to validate.

    Returns:
        List of ValidationIssue (empty if valid).
    """
    from pipeline.formatters.input_validator import InputValidator

    issues: List[ValidationIssue] = []
    validator = InputValidator()
    result = validator.validate_content(data)

    for err_msg in result.errors:
        issues.append(ValidationIssue(
            level="error",
            field=_extract_field_from_message(err_msg),
            message=err_msg,
        ))

    for warn_msg in result.warnings:
        issues.append(ValidationIssue(
            level="info",
            field="enrichments",
            message=warn_msg,
        ))

    return issues


def validate_format(data: Dict[str, Any]) -> List[ValidationIssue]:
    """Validate data against FormatV1 schema.

    Args:
        data: Parsed JSON data to validate.

    Returns:
        List of ValidationIssue (empty if valid).
    """
    from pipeline.formatters.schemas.format_v1 import FormatV1

    issues: List[ValidationIssue] = []
    try:
        FormatV1(**data)
    except ValidationError as e:
        issues.extend(_pydantic_errors_to_issues(e.errors(), SCHEMA_FORMAT))
    except Exception as e:
        issues.append(ValidationIssue(
            level="error",
            field="root",
            message=f"Unexpected validation error: {e}",
        ))
    return issues


def detect_schema_type(data: Dict[str, Any]) -> Optional[str]:
    """Auto-detect schema type from file content.

    Heuristics:
    - Has 'enrichment_version' → enrichment
    - Has 'format_version' → format
    - Has 'metadata.schema_version' starting with 'transcript' → transcript
    - Has 'transcript' key with list of segments → transcript

    Args:
        data: Parsed JSON data.

    Returns:
        Schema type string or None if undetectable.
    """
    if "enrichment_version" in data:
        return SCHEMA_ENRICHMENT
    if "format_version" in data:
        return SCHEMA_FORMAT
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        sv = metadata.get("schema_version", "")
        if isinstance(sv, str) and sv.startswith("transcript"):
            return SCHEMA_TRANSCRIPT
    if "transcript" in data and isinstance(data["transcript"], list):
        return SCHEMA_TRANSCRIPT
    return None


def validate_by_schema(
    data: Dict[str, Any],
    schema_type: str,
) -> List[ValidationIssue]:
    """Validate data against the specified schema type.

    Args:
        data: Parsed JSON data.
        schema_type: One of 'transcript', 'enrichment', 'format'.

    Returns:
        List of ValidationIssue.

    Raises:
        ValueError: If schema_type is invalid.
    """
    validators = {
        SCHEMA_TRANSCRIPT: validate_transcript,
        SCHEMA_ENRICHMENT: validate_enrichment,
        SCHEMA_FORMAT: validate_format,
    }
    if schema_type not in validators:
        raise ValueError(
            f"Unknown schema type: {schema_type}. "
            f"Valid types: {VALID_SCHEMA_TYPES}"
        )
    return validators[schema_type](data)


def _extract_field_from_message(msg: str) -> str:
    """Extract field name from an error message like 'Missing required field: X'."""
    if "field:" in msg.lower():
        parts = msg.split(":")
        if len(parts) >= 2:
            return parts[-1].strip()
    if "enrichment version" in msg.lower():
        return "enrichment_version"
    return "root"
