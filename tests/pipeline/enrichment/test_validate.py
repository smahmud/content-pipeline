"""
Unit tests for Schema Validation and Repair

Tests validation, JSON extraction, repair logic, and error handling.
"""

import pytest
import json
from pydantic import ValidationError

from pipeline.enrichment.validate import (
    SchemaValidator,
    validate_enrichment_response,
    validate_and_repair_enrichment
)
from pipeline.enrichment.schemas import (
    SummaryEnrichment,
    TagEnrichment,
    ChapterEnrichment,
    HighlightEnrichment
)
from pipeline.enrichment.errors import SchemaValidationError


@pytest.fixture
def validator():
    """Create schema validator."""
    return SchemaValidator()


class TestSchemaValidator:
    """Test suite for SchemaValidator."""
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.schema_map is not None
        assert "summary" in validator.schema_map
        assert "tag" in validator.schema_map
        assert "chapter" in validator.schema_map
        assert "highlight" in validator.schema_map
    
    def test_validate_response_valid_summary(self, validator):
        """Test validation of valid summary response."""
        response = json.dumps({
            "short": "This is a short summary.",
            "medium": "This is a medium-length summary with more details.",
            "long": "This is a long summary with comprehensive details about the content."
        })
        
        result = validator.validate_response(response, "summary")
        
        assert isinstance(result, SummaryEnrichment)
        assert result.short == "This is a short summary."
    
    def test_validate_response_valid_tag(self, validator):
        """Test validation of valid tag response."""
        response = json.dumps({
            "categories": ["technology", "ai"],
            "keywords": ["machine learning", "neural networks"],
            "entities": ["OpenAI", "Google"]
        })
        
        result = validator.validate_response(response, "tag")
        
        assert isinstance(result, TagEnrichment)
        assert "technology" in result.categories
        assert "machine learning" in result.keywords
    
    def test_validate_response_invalid_enrichment_type(self, validator):
        """Test validation with invalid enrichment type."""
        response = json.dumps({"test": "data"})
        
        with pytest.raises(ValueError) as exc_info:
            validator.validate_response(response, "invalid_type")
        
        assert "Unknown enrichment type" in str(exc_info.value)
    
    def test_validate_response_invalid_json(self, validator):
        """Test validation with invalid JSON."""
        response = "This is not JSON"
        
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_response(response, "summary", attempt_repair=False)
        
        assert "Failed to parse JSON" in str(exc_info.value)
    
    def test_validate_response_missing_required_field(self, validator):
        """Test validation with missing required field."""
        response = json.dumps({
            "short": "Short summary",
            "medium": "Medium summary"
            # Missing 'long' field
        })
        
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_response(response, "summary", attempt_repair=False)
        
        assert "Schema validation failed" in str(exc_info.value)
    
    def test_extract_json_direct(self, validator):
        """Test JSON extraction from direct JSON string."""
        json_str = '{"key": "value"}'
        
        data = validator._extract_json(json_str)
        
        assert data == {"key": "value"}
    
    def test_extract_json_from_markdown(self, validator):
        """Test JSON extraction from markdown code block."""
        text = """
        Here is the JSON:
        ```json
        {"key": "value"}
        ```
        """
        
        data = validator._extract_json(text)
        
        assert data == {"key": "value"}
    
    def test_extract_json_from_text(self, validator):
        """Test JSON extraction from text with embedded JSON."""
        text = "Some text before {\"key\": \"value\"} some text after"
        
        data = validator._extract_json(text)
        
        assert data == {"key": "value"}
    
    def test_repair_json_single_quotes(self, validator):
        """Test JSON repair for single quotes."""
        malformed = "{'key': 'value'}"
        
        data = validator._repair_json(malformed)
        
        assert data == {"key": "value"}
    
    def test_repair_json_trailing_comma(self, validator):
        """Test JSON repair for trailing commas."""
        malformed = '{"key": "value",}'
        
        data = validator._repair_json(malformed)
        
        assert data == {"key": "value"}
    
    def test_repair_data_string_too_long(self, validator):
        """Test data repair for string too long."""
        data = {
            "short": "x" * 1000,  # Exceeds max_length of 500
            "medium": "Medium summary",
            "long": "Long summary"
        }
        
        # Create mock validation error
        error = ValidationError.from_exception_data(
            "SummaryEnrichment",
            [
                {
                    "type": "string_too_long",
                    "loc": ("short",),
                    "msg": "String too long",
                    "ctx": {"max_length": 500}
                }
            ]
        )
        
        repaired = validator._repair_data(data, SummaryEnrichment, error)
        
        assert repaired is not None
        assert len(repaired["short"]) == 500
    
    def test_repair_data_missing_optional_field(self, validator):
        """Test data repair for missing optional field."""
        data = {
            "timestamp": "00:10:00",
            "quote": "Important quote",
            "importance": "high"
            # Missing optional 'context' field
        }
        
        # Create mock validation error
        error = ValidationError.from_exception_data(
            "HighlightEnrichment",
            [
                {
                    "type": "missing",
                    "loc": ("context",),
                    "msg": "Field required"
                }
            ]
        )
        
        repaired = validator._repair_data(data, HighlightEnrichment, error)
        
        assert repaired is not None
        assert repaired["context"] is None
    
    def test_fix_timestamp_valid(self, validator):
        """Test timestamp fixing for valid format."""
        timestamp = "01:23:45"
        
        fixed = validator._fix_timestamp(timestamp)
        
        assert fixed == "01:23:45"
    
    def test_fix_timestamp_mm_ss(self, validator):
        """Test timestamp fixing for MM:SS format."""
        timestamp = "23:45"
        
        fixed = validator._fix_timestamp(timestamp)
        
        assert fixed == "00:23:45"
    
    def test_fix_timestamp_seconds_only(self, validator):
        """Test timestamp fixing for seconds only."""
        timestamp = "125"  # 2 minutes 5 seconds
        
        fixed = validator._fix_timestamp(timestamp)
        
        assert fixed == "00:02:05"
    
    def test_fix_timestamp_with_extra_chars(self, validator):
        """Test timestamp fixing with extra characters."""
        timestamp = "at 01:23:45 mark"
        
        fixed = validator._fix_timestamp(timestamp)
        
        assert fixed == "01:23:45"
    
    def test_fix_timestamp_invalid(self, validator):
        """Test timestamp fixing for invalid format."""
        timestamp = "invalid"
        
        fixed = validator._fix_timestamp(timestamp)
        
        assert fixed is None
    
    def test_validate_response_with_repair(self, validator):
        """Test validation with automatic repair."""
        # Malformed JSON with single quotes
        response = "{'short': 'Short', 'medium': 'Medium', 'long': 'Long'}"
        
        result = validator.validate_response(response, "summary", attempt_repair=True)
        
        assert isinstance(result, SummaryEnrichment)
        assert result.short == "Short"
    
    def test_validate_enrichment_response_convenience(self):
        """Test convenience function for validation."""
        response = json.dumps({
            "short": "Short summary",
            "medium": "Medium summary",
            "long": "Long summary"
        })
        
        result = validate_enrichment_response(response, "summary")
        
        assert isinstance(result, SummaryEnrichment)
    
    def test_validate_and_repair_enrichment_convenience(self):
        """Test convenience function for validation and repair."""
        response = json.dumps({
            "short": "Short summary",
            "medium": "Medium summary",
            "long": "Long summary"
        })
        
        result = validate_and_repair_enrichment(response, "summary")
        
        # Should return JSON string
        assert isinstance(result, str)
        
        # Should be valid JSON
        data = json.loads(result)
        assert "short" in data
        assert "medium" in data
        assert "long" in data
    
    def test_schema_validation_error_attributes(self):
        """Test SchemaValidationError attributes."""
        try:
            validator = SchemaValidator()
            validator.validate_response("invalid", "summary", attempt_repair=False)
        except SchemaValidationError as e:
            assert e.enrichment_type == "summary"
            assert e.response_text == "invalid"
            assert e.original_error is not None
    
    def test_validate_chapter_enrichment(self, validator):
        """Test validation of chapter enrichment."""
        response = json.dumps([{
            "title": "Introduction",
            "start_time": "00:00:00",
            "end_time": "00:05:00",
            "description": "Opening remarks and introduction to the topic."
        }])
        
        result = validator.validate_response(response, "chapter")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ChapterEnrichment)
        assert result[0].title == "Introduction"
        assert result[0].start_time == "00:00:00"
    
    def test_validate_highlight_enrichment(self, validator):
        """Test validation of highlight enrichment."""
        response = json.dumps([{
            "timestamp": "00:10:30",
            "quote": "This is a key insight from the discussion.",
            "importance": "high",
            "context": "During the main topic discussion"
        }])
        
        result = validator.validate_response(response, "highlight")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], HighlightEnrichment)
        assert result[0].timestamp == "00:10:30"
        assert result[0].importance == "high"
    
    def test_repair_data_cannot_fix(self, validator):
        """Test data repair when repair is not possible."""
        data = {
            # Missing all required fields
        }
        
        # Create mock validation error
        error = ValidationError.from_exception_data(
            "SummaryEnrichment",
            [
                {
                    "type": "missing",
                    "loc": ("short",),
                    "msg": "Field required"
                }
            ]
        )
        
        repaired = validator._repair_data(data, SummaryEnrichment, error)
        
        # Should return None when repair is not possible
        assert repaired is None
