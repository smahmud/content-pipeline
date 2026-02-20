"""Validate tool — wraps the validate CLI command."""

from typing import Optional

from pipeline.validation.engine import ValidationEngine


async def validate(
    input_path: str,
    schema: str = "auto",
    platform: Optional[str] = None,
    strict: bool = False,
) -> dict:
    """Validate pipeline artifacts against schemas and platform limits.

    Args:
        input_path: Path to JSON file to validate.
        schema: Schema type (auto, transcript, enrichment, format).
        platform: Optional platform to validate against.
        strict: Fail on warnings in addition to errors.

    Returns:
        Dict with validation results.
    """
    try:
        engine = ValidationEngine(strict=strict)
        report = engine.validate_file(input_path, schema, platform)

        return {
            "success": True,
            "is_valid": report.is_valid,
            "schema_type": report.schema_type,
            "file_path": report.file_path,
            "issues": [i.to_dict() for i in report.issues],
            "summary": {
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "infos": len(report.infos),
            },
            "duration_ms": report.duration_ms,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
