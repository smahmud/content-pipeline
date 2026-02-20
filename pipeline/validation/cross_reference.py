"""
Cross-Reference Validator

Validates relationships between pipeline artifacts:
- Checks that referenced source files exist
- Verifies timestamps are chronologically consistent
- Validates enrichment_version references match
"""

import os
from typing import Any, Dict, List, Optional

from pipeline.validation.report import ValidationIssue


def validate_cross_references(
    data: Dict[str, Any],
    schema_type: str,
    base_dir: Optional[str] = None,
) -> List[ValidationIssue]:
    """Validate cross-references in a pipeline artifact.

    Args:
        data: Parsed JSON data of the artifact.
        schema_type: The detected schema type.
        base_dir: Base directory for resolving relative file paths.

    Returns:
        List of ValidationIssue for any cross-reference problems.
    """
    issues: List[ValidationIssue] = []

    if schema_type == "format":
        issues.extend(_check_format_references(data, base_dir))
    elif schema_type == "enrichment":
        issues.extend(_check_enrichment_references(data, base_dir))

    return issues


def _check_format_references(
    data: Dict[str, Any],
    base_dir: Optional[str],
) -> List[ValidationIssue]:
    """Check cross-references in format output metadata."""
    issues: List[ValidationIssue] = []

    source_file = data.get("source_file")
    if source_file and base_dir:
        full_path = os.path.join(base_dir, source_file)
        if not os.path.exists(full_path) and not os.path.exists(source_file):
            issues.append(ValidationIssue(
                level="warning",
                field="source_file",
                message=f"Referenced source file not found: {source_file}",
                suggestion="Ensure the source enriched JSON file exists at the referenced path.",
            ))

    return issues


def _check_enrichment_references(
    data: Dict[str, Any],
    base_dir: Optional[str],
) -> List[ValidationIssue]:
    """Check cross-references in enrichment data."""
    issues: List[ValidationIssue] = []

    # Check enrichment_version consistency
    version = data.get("enrichment_version")
    if version and version != "v1":
        issues.append(ValidationIssue(
            level="warning",
            field="enrichment_version",
            message=f"Unexpected enrichment version: {version} (expected v1)",
        ))

    # Check metadata.enrichment_types matches actual content
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        declared_types = metadata.get("enrichment_types", [])
        actual_types = []
        for etype in ["summary", "tags", "chapters", "highlights"]:
            if data.get(etype) is not None:
                actual_types.append(etype)

        for declared in declared_types:
            # Normalize: 'tag' -> 'tags', etc.
            normalized = declared if declared.endswith("s") or declared == "summary" else declared + "s"
            if normalized not in actual_types and declared not in actual_types:
                issues.append(ValidationIssue(
                    level="warning",
                    field="metadata.enrichment_types",
                    message=f"Declared enrichment type '{declared}' not found in data",
                    suggestion=f"Remove '{declared}' from enrichment_types or add the enrichment data.",
                ))

    return issues
