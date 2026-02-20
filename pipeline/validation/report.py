"""
Validation Report Data Models

Defines ValidationIssue and ValidationReport dataclasses used across
the validation module for structured error/warning reporting.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional


@dataclass
class ValidationIssue:
    """A single validation issue found during validation.

    Attributes:
        level: Severity level (error, warning, info)
        field: The field or path where the issue was found
        message: Human-readable description of the issue
        suggestion: Optional suggestion for fixing the issue
    """
    level: Literal["error", "warning", "info"]
    field: str
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {"level": self.level, "field": self.field, "message": self.message}
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class ValidationReport:
    """Structured report from a validation run.

    Attributes:
        file_path: Path to the validated file
        schema_type: Detected or specified schema type
        is_valid: Whether the file passed validation
        issues: List of validation issues found
        timestamp: When validation was performed (UTC)
        duration_ms: How long validation took in milliseconds
    """
    file_path: str
    schema_type: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: int = 0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "info"]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "schema_type": self.schema_type,
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def format_human(self) -> str:
        """Format report for human-readable console output."""
        if self.is_valid and not self.warnings:
            icon = "✅"
            status = "Valid"
        elif self.is_valid and self.warnings:
            icon = "✅"
            status = "Valid (with warnings)"
        else:
            icon = "❌"
            status = "Failed"

        lines = [f"{icon} {self.file_path}: {status} ({self.schema_type})"]

        for issue in self.issues:
            if issue.level == "error":
                prefix = "  ❌"
            elif issue.level == "warning":
                prefix = "  ⚠"
            else:
                prefix = "  ℹ"
            lines.append(f"{prefix} [{issue.field}] {issue.message}")
            if issue.suggestion:
                lines.append(f"      → {issue.suggestion}")

        return "\n".join(lines)
