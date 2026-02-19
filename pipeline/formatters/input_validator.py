"""
Input Validation Module for Formatter

Validates that input files conform to the EnrichmentV1 schema and contain
all required enrichment fields for the requested output type.

This module provides:
- EnrichmentV1 schema validation
- Required field checking per output type
- Clear error messages for missing fields
- Suggestions for running enrichment if fields are missing

Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline.formatters.base import VALID_OUTPUT_TYPES, OutputType
from pipeline.formatters.errors import InputValidationError


# Required enrichments per output type
# Maps output type to list of required enrichment field paths
REQUIRED_ENRICHMENTS: Dict[str, List[str]] = {
    # Tier 1 - Core formats
    OutputType.BLOG.value: ["summary", "tags"],
    OutputType.TWEET.value: ["summary"],
    OutputType.YOUTUBE.value: ["summary"],
    OutputType.SEO.value: ["summary", "tags"],
    
    # Tier 2 - Extended formats
    OutputType.LINKEDIN.value: ["summary"],
    OutputType.NEWSLETTER.value: ["summary"],
    OutputType.CHAPTERS.value: ["chapters"],
    OutputType.TRANSCRIPT_CLEAN.value: ["transcript"],
    
    # Tier 3 - Specialized formats
    OutputType.PODCAST_NOTES.value: ["summary"],
    OutputType.MEETING_MINUTES.value: ["summary"],
    OutputType.SLIDES.value: ["summary"],
    OutputType.NOTION.value: ["summary"],
    OutputType.OBSIDIAN.value: ["summary"],
    OutputType.QUOTE_CARDS.value: ["highlights"],
    
    # AI Video formats
    OutputType.VIDEO_SCRIPT.value: ["summary"],
    OutputType.TIKTOK_SCRIPT.value: ["summary"],
}


# Required fields in EnrichmentV1 schema
ENRICHMENT_V1_REQUIRED_FIELDS = [
    "enrichment_version",
    "metadata",
]

# Required fields in metadata
METADATA_REQUIRED_FIELDS = [
    "provider",
    "model",
    "timestamp",
    "cost_usd",
    "tokens_used",
    "enrichment_types",
]


@dataclass
class ValidationResult:
    """Result of input validation.
    
    Attributes:
        is_valid: Whether the input is valid
        errors: List of validation error messages
        warnings: List of validation warnings
        available_enrichments: List of enrichment types available in the input
        missing_enrichments: List of required enrichments that are missing
        suggestions: List of suggestions for fixing validation errors
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    available_enrichments: List[str] = field(default_factory=list)
    missing_enrichments: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class InputValidator:
    """Validates input files for the formatter.
    
    Provides comprehensive validation of enriched JSON files including:
    - File existence and readability
    - JSON parsing
    - EnrichmentV1 schema conformance
    - Required enrichment fields for output type
    """
    
    def validate_file(
        self,
        file_path: Union[str, Path],
        output_type: Optional[str] = None,
    ) -> ValidationResult:
        """Validate an input file.
        
        Args:
            file_path: Path to the enriched JSON file
            output_type: Optional output type to check required enrichments
            
        Returns:
            ValidationResult with validation status and details
        """
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"Input file not found: {file_path}"],
                suggestions=["Check that the file path is correct."],
            )
        
        # Check file is readable
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                errors=[f"Path is not a file: {file_path}"],
                suggestions=["Provide a path to a file, not a directory."],
            )
        
        # Read and parse JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {e}"],
                suggestions=["Check that the file contains valid JSON."],
            )
        except IOError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Cannot read file: {e}"],
                suggestions=["Check file permissions."],
            )
        
        # Validate content
        return self.validate_content(data, output_type)
    
    def validate_content(
        self,
        data: Dict[str, Any],
        output_type: Optional[str] = None,
    ) -> ValidationResult:
        """Validate enriched content data.
        
        Args:
            data: Enriched content dictionary
            output_type: Optional output type to check required enrichments
            
        Returns:
            ValidationResult with validation status and details
        """
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []
        
        # Validate EnrichmentV1 schema
        schema_errors = self._validate_schema(data)
        errors.extend(schema_errors)
        
        # Get available enrichments
        available = self._get_available_enrichments(data)
        
        # Check required enrichments for output type
        missing: List[str] = []
        if output_type:
            if output_type not in VALID_OUTPUT_TYPES:
                errors.append(f"Invalid output type: {output_type}")
                suggestions.append(f"Valid output types: {', '.join(VALID_OUTPUT_TYPES)}")
            else:
                required = REQUIRED_ENRICHMENTS.get(output_type, [])
                missing = self._check_required_enrichments(data, required)
                
                if missing:
                    errors.extend([
                        f"Missing required enrichment for {output_type}: {field}"
                        for field in missing
                    ])
                    suggestions.append(
                        f"Run enrichment with --types {','.join(missing)} to add missing fields."
                    )
        
        # Add info about available enrichments
        if available:
            info_msg = f"Available enrichments: {', '.join(available)}"
            if not errors:
                warnings.append(info_msg)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            available_enrichments=available,
            missing_enrichments=missing,
            suggestions=suggestions,
        )
    
    def _validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate EnrichmentV1 schema structure.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            List of schema validation errors
        """
        errors: List[str] = []
        
        # Check required top-level fields
        for field_name in ENRICHMENT_V1_REQUIRED_FIELDS:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")
        
        # Check enrichment_version
        if "enrichment_version" in data:
            version = data["enrichment_version"]
            if version != "v1":
                errors.append(
                    f"Unsupported enrichment version: {version}. Expected: v1"
                )
        
        # Check metadata structure
        if "metadata" in data:
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                errors.append("Field 'metadata' must be an object")
            else:
                for field_name in METADATA_REQUIRED_FIELDS:
                    if field_name not in metadata:
                        errors.append(f"Missing required metadata field: {field_name}")
        
        # Check that at least one enrichment type is present
        enrichment_fields = ["summary", "tags", "chapters", "highlights"]
        has_enrichment = any(
            field in data and data[field] is not None
            for field in enrichment_fields
        )
        
        if not has_enrichment:
            errors.append(
                "At least one enrichment type must be present "
                "(summary, tags, chapters, or highlights)"
            )
        
        return errors
    
    def _get_available_enrichments(self, data: Dict[str, Any]) -> List[str]:
        """Get list of available enrichment types in the data.
        
        Args:
            data: Enriched content dictionary
            
        Returns:
            List of available enrichment type names
        """
        available: List[str] = []
        
        # Check standard enrichment fields
        enrichment_fields = {
            "summary": "summary",
            "tags": "tags",
            "chapters": "chapters",
            "highlights": "highlights",
        }
        
        for field_name, display_name in enrichment_fields.items():
            if self._has_field(data, field_name):
                available.append(display_name)
        
        # Check for transcript (special case - may be in different location)
        if self._has_field(data, "transcript"):
            available.append("transcript")
        
        return available
    
    def _check_required_enrichments(
        self,
        data: Dict[str, Any],
        required: List[str],
    ) -> List[str]:
        """Check which required enrichments are missing.
        
        Args:
            data: Enriched content dictionary
            required: List of required enrichment field names
            
        Returns:
            List of missing enrichment field names
        """
        missing: List[str] = []
        
        for field_name in required:
            if not self._has_field(data, field_name):
                missing.append(field_name)
        
        return missing
    
    def _has_field(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a field exists and has a non-empty value.
        
        Args:
            data: Dictionary to check
            field_path: Dot-separated path (e.g., "summary.short")
            
        Returns:
            True if field exists and is not None/empty
        """
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if not isinstance(current, dict):
                return False
            if part not in current:
                return False
            current = current[part]
        
        # Check if value is not None/empty
        if current is None:
            return False
        if isinstance(current, str) and not current.strip():
            return False
        if isinstance(current, (list, dict)) and len(current) == 0:
            return False
        
        return True
    
    def get_required_enrichments(self, output_type: str) -> List[str]:
        """Get the required enrichments for an output type.
        
        Args:
            output_type: The output type to get requirements for
            
        Returns:
            List of required enrichment field names
            
        Raises:
            ValueError: If output_type is invalid
        """
        if output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                f"Invalid output type: {output_type}. "
                f"Valid types: {', '.join(VALID_OUTPUT_TYPES)}"
            )
        
        return REQUIRED_ENRICHMENTS.get(output_type, [])


def validate_input(
    data: Dict[str, Any],
    output_type: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Convenience function to validate input data.
    
    Args:
        data: Enriched content dictionary
        output_type: Optional output type to check required enrichments
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    validator = InputValidator()
    result = validator.validate_content(data, output_type)
    return result.is_valid, result.errors


def validate_input_file(
    file_path: Union[str, Path],
    output_type: Optional[str] = None,
) -> ValidationResult:
    """Convenience function to validate an input file.
    
    Args:
        file_path: Path to the enriched JSON file
        output_type: Optional output type to check required enrichments
        
    Returns:
        ValidationResult with validation status and details
    """
    validator = InputValidator()
    return validator.validate_file(file_path, output_type)


def get_required_enrichments_for_type(output_type: str) -> List[str]:
    """Get the required enrichments for an output type.
    
    Args:
        output_type: The output type to get requirements for
        
    Returns:
        List of required enrichment field names
        
    Raises:
        ValueError: If output_type is invalid
    """
    validator = InputValidator()
    return validator.get_required_enrichments(output_type)
