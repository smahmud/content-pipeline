"""
Schema Validation and Repair

Validates LLM responses against enrichment schemas and attempts to repair
malformed responses when possible.
"""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

from pipeline.enrichment.schemas import (
    SummaryEnrichment,
    TagEnrichment,
    ChapterEnrichment,
    HighlightEnrichment,
    EnrichmentV1,
)
from pipeline.enrichment.errors import SchemaValidationError

T = TypeVar('T', bound=BaseModel)


class SchemaValidator:
    """Validates and repairs LLM responses against enrichment schemas."""
    
    def __init__(self):
        """Initialize the schema validator."""
        self.schema_map = {
            'summary': SummaryEnrichment,
            'tag': TagEnrichment,
            'chapter': ChapterEnrichment,
            'highlight': HighlightEnrichment,
        }
    
    def validate_response(
        self,
        response_text: str,
        enrichment_type: str,
        attempt_repair: bool = True
    ) -> Union[SummaryEnrichment, TagEnrichment, ChapterEnrichment, HighlightEnrichment]:
        """Validate LLM response against the appropriate schema.
        
        Args:
            response_text: Raw text response from LLM
            enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
            attempt_repair: Whether to attempt repairing malformed responses
            
        Returns:
            Validated enrichment object
            
        Raises:
            SchemaValidationError: If validation fails and repair is unsuccessful
        """
        if enrichment_type not in self.schema_map:
            raise ValueError(
                f"Unknown enrichment type: {enrichment_type}. "
                f"Must be one of: {list(self.schema_map.keys())}"
            )
        
        schema_class = self.schema_map[enrichment_type]
        
        # Try to parse JSON from response
        try:
            data = self._extract_json(response_text)
        except json.JSONDecodeError as e:
            if attempt_repair:
                # Try to repair JSON
                data = self._repair_json(response_text)
                if data is None:
                    raise SchemaValidationError(
                        f"Failed to parse JSON from LLM response: {e}",
                        enrichment_type=enrichment_type,
                        response_text=response_text,
                        original_error=e
                    )
            else:
                raise SchemaValidationError(
                    f"Failed to parse JSON from LLM response: {e}",
                    enrichment_type=enrichment_type,
                    response_text=response_text,
                    original_error=e
                )
        
        # Try to validate against schema
        try:
            return schema_class(**data)
        except ValidationError as e:
            if attempt_repair:
                # Try to repair data
                repaired_data = self._repair_data(data, schema_class, e)
                if repaired_data is not None:
                    try:
                        return schema_class(**repaired_data)
                    except ValidationError as e2:
                        raise SchemaValidationError(
                            f"Schema validation failed even after repair: {e2}",
                            enrichment_type=enrichment_type,
                            response_text=response_text,
                            original_error=e
                        )
            
            raise SchemaValidationError(
                f"Schema validation failed: {e}",
                enrichment_type=enrichment_type,
                response_text=response_text,
                original_error=e
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain markdown or other formatting.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON data
            
        Raises:
            json.JSONDecodeError: If JSON cannot be extracted
        """
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If all else fails, raise error
        raise json.JSONDecodeError("No valid JSON found in text", text, 0)
    
    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair malformed JSON.
        
        Args:
            text: Malformed JSON text
            
        Returns:
            Repaired JSON data, or None if repair failed
        """
        # Common JSON issues and fixes
        repairs = [
            # Fix single quotes to double quotes
            (r"'", '"'),
            # Fix trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix missing quotes around keys
            (r'(\w+):', r'"\1":'),
        ]
        
        repaired_text = text
        for pattern, replacement in repairs:
            repaired_text = re.sub(pattern, replacement, repaired_text)
        
        try:
            return self._extract_json(repaired_text)
        except json.JSONDecodeError:
            return None
    
    def _repair_data(
        self,
        data: Dict[str, Any],
        schema_class: Type[T],
        validation_error: ValidationError
    ) -> Optional[Dict[str, Any]]:
        """Attempt to repair data that failed schema validation.
        
        Args:
            data: Data that failed validation
            schema_class: Schema class to validate against
            validation_error: The validation error that occurred
            
        Returns:
            Repaired data, or None if repair failed
        """
        repaired_data = data.copy()
        
        # Extract error details
        errors = validation_error.errors()
        
        for error in errors:
            field = error['loc'][0] if error['loc'] else None
            error_type = error['type']
            
            if not field:
                continue
            
            # Handle missing required fields
            if error_type == 'missing':
                # Try to provide default values
                if field == 'categories' or field == 'keywords' or field == 'entities':
                    repaired_data[field] = []
                elif field == 'context':
                    repaired_data[field] = None
                else:
                    # Can't repair missing required field
                    return None
            
            # Handle string too long
            elif error_type == 'string_too_long':
                if isinstance(repaired_data.get(field), str):
                    max_length = error.get('ctx', {}).get('max_length')
                    if max_length:
                        repaired_data[field] = repaired_data[field][:max_length]
            
            # Handle list too short
            elif error_type == 'too_short':
                if isinstance(repaired_data.get(field), list):
                    min_items = error.get('ctx', {}).get('min_length', 1)
                    if len(repaired_data[field]) < min_items:
                        # Can't repair - need actual data
                        return None
            
            # Handle invalid timestamp format
            elif error_type == 'value_error' and field in ['timestamp', 'start_time', 'end_time']:
                # Try to fix common timestamp issues
                timestamp = repaired_data.get(field, '')
                fixed_timestamp = self._fix_timestamp(timestamp)
                if fixed_timestamp:
                    repaired_data[field] = fixed_timestamp
                else:
                    return None
        
        return repaired_data
    
    def _fix_timestamp(self, timestamp: str) -> Optional[str]:
        """Attempt to fix malformed timestamp.
        
        Args:
            timestamp: Malformed timestamp string
            
        Returns:
            Fixed timestamp in HH:MM:SS format, or None if unfixable
        """
        # Remove any non-digit, non-colon characters
        cleaned = re.sub(r'[^\d:]', '', timestamp)
        
        # Try to parse as HH:MM:SS
        parts = cleaned.split(':')
        if len(parts) == 3:
            try:
                h, m, s = map(int, parts)
                # Validate ranges
                if 0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59:
                    return f"{h:02d}:{m:02d}:{s:02d}"
            except ValueError:
                pass
        
        # Try to parse as MM:SS (assume 00 hours)
        elif len(parts) == 2:
            try:
                m, s = map(int, parts)
                if 0 <= m <= 59 and 0 <= s <= 59:
                    return f"00:{m:02d}:{s:02d}"
            except ValueError:
                pass
        
        # Try to parse as seconds only
        elif len(parts) == 1:
            try:
                total_seconds = int(parts[0])
                h = total_seconds // 3600
                m = (total_seconds % 3600) // 60
                s = total_seconds % 60
                if h <= 99:
                    return f"{h:02d}:{m:02d}:{s:02d}"
            except ValueError:
                pass
        
        return None


def validate_enrichment_response(
    response_text: str,
    enrichment_type: str,
    attempt_repair: bool = True
) -> Union[SummaryEnrichment, TagEnrichment, ChapterEnrichment, HighlightEnrichment]:
    """Convenience function to validate an enrichment response.
    
    Args:
        response_text: Raw text response from LLM
        enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
        attempt_repair: Whether to attempt repairing malformed responses
        
    Returns:
        Validated enrichment object
        
    Raises:
        SchemaValidationError: If validation fails
    """
    validator = SchemaValidator()
    return validator.validate_response(response_text, enrichment_type, attempt_repair)


def validate_and_repair_enrichment(
    response_text: str,
    enrichment_type: str
) -> str:
    """Validate and repair enrichment response, returning JSON string.
    
    This is a convenience function for the orchestrator that validates
    the response and returns it as a JSON string.
    
    Args:
        response_text: Raw text response from LLM
        enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
        
    Returns:
        Validated and repaired JSON string
        
    Raises:
        SchemaValidationError: If validation fails
    """
    validated = validate_enrichment_response(response_text, enrichment_type, attempt_repair=True)
    return json.dumps(validated.model_dump())
