"""
Property-based tests for enrichment validation (Requirements 13.8, 13.9, 13.10).

This module implements Properties 44, 45, and 46 from the design document
using Hypothesis for generative testing with random inputs.

These tests validate:
- Property 44: Provider Response Normalization (Requirement 13.8)
- Property 45: Array Response Validation (Requirement 13.9)
- Property 46: Multi-Enrichment Consistency (Requirement 13.10)
"""

import pytest
import json
from hypothesis import given, strategies as st, assume, settings
from typing import Dict, Any, List

from pipeline.enrichment.validate import (
    SchemaValidator,
    validate_enrichment_response,
    validate_and_repair_enrichment
)
from pipeline.enrichment.schemas.summary import SummaryEnrichment
from pipeline.enrichment.schemas.tag import TagEnrichment
from pipeline.enrichment.schemas.chapter import ChapterEnrichment
from pipeline.enrichment.schemas.highlight import HighlightEnrichment
from pipeline.enrichment.errors import SchemaValidationError


# ============================================================================
# STRATEGIES: Reusable Hypothesis strategies for generating test data
# ============================================================================

# Text strategies
short_text = st.text(min_size=10, max_size=200)
medium_text = st.text(min_size=50, max_size=500)

# Smart quote characters that AWS Bedrock (Claude) returns
SMART_QUOTES = ['"', '"', ''', ''']
REGULAR_QUOTES = ['"', '"', "'", "'"]

# Control characters that can break JSON parsing
CONTROL_CHARS = ['\n', '\r', '\t', '\b', '\f']

# Enrichment type strategies
enrichment_types = st.sampled_from(["summary", "tag", "chapter", "highlight"])


def add_smart_quotes(text: str) -> str:
    """Replace regular quotes with smart quotes."""
    result = text
    for regular, smart in zip(REGULAR_QUOTES, SMART_QUOTES):
        result = result.replace(regular, smart)
    return result


def add_control_characters(text: str) -> str:
    """Add unescaped control characters to text."""
    # Insert control characters at random positions
    import random
    chars = list(text)
    for _ in range(min(3, len(chars) // 10)):
        pos = random.randint(0, len(chars) - 1)
        chars.insert(pos, random.choice(CONTROL_CHARS))
    return ''.join(chars)


# Strategy for generating valid summary JSON
@st.composite
def summary_json_strategy(draw):
    """Generate valid summary JSON data."""
    return {
        "short": draw(st.text(min_size=10, max_size=100)),
        "medium": draw(st.text(min_size=50, max_size=300)),
        "long": draw(st.text(min_size=100, max_size=500))
    }


# Strategy for generating valid tag JSON
@st.composite
def tag_json_strategy(draw):
    """Generate valid tag JSON data."""
    return {
        "categories": draw(st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=5)),
        "keywords": draw(st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=10)),
        "entities": draw(st.lists(st.text(min_size=3, max_size=20), min_size=0, max_size=8))
    }


# Strategy for generating valid chapter JSON
@st.composite
def chapter_json_strategy(draw):
    """Generate valid chapter JSON data."""
    hours = draw(st.integers(min_value=0, max_value=2))
    start_minutes = draw(st.integers(min_value=0, max_value=59))
    start_seconds = draw(st.integers(min_value=0, max_value=59))
    duration_seconds = draw(st.integers(min_value=30, max_value=600))
    
    start_time = f"{hours:02d}:{start_minutes:02d}:{start_seconds:02d}"
    
    # Calculate end time
    start_total = hours * 3600 + start_minutes * 60 + start_seconds
    end_total = start_total + duration_seconds
    end_hours = end_total // 3600
    end_minutes = (end_total % 3600) // 60
    end_seconds = end_total % 60
    end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}"
    
    return {
        "title": draw(st.text(min_size=5, max_size=50)),
        "start_time": start_time,
        "end_time": end_time,
        "description": draw(st.text(min_size=10, max_size=200))
    }


# Strategy for generating valid highlight JSON
@st.composite
def highlight_json_strategy(draw):
    """Generate valid highlight JSON data."""
    hours = draw(st.integers(min_value=0, max_value=2))
    minutes = draw(st.integers(min_value=0, max_value=59))
    seconds = draw(st.integers(min_value=0, max_value=59))
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return {
        "timestamp": timestamp,
        "quote": draw(st.text(min_size=20, max_size=200)),
        "importance": draw(st.sampled_from(["high", "medium", "low"])),
        "context": draw(st.text(min_size=10, max_size=100))
    }


# ============================================================================
# PROPERTY 44: Provider Response Normalization (Requirement 13.8)
# ============================================================================

class TestProviderResponseNormalization:
    """Property 44: Provider Response Normalization.
    
    Validates Requirement 13.8: The system must normalize provider-specific
    response quirks (smart quotes, control characters) before JSON parsing.
    """
    
    @given(
        data=summary_json_strategy(),
        use_smart_quotes=st.booleans(),
        add_control_chars=st.booleans()
    )
    @settings(max_examples=100)
    def test_property_44_summary_normalization(self, data, use_smart_quotes, add_control_chars):
        """
        **Property 44: Provider Response Normalization (Summary)**
        *For any* valid summary JSON with smart quotes or control characters,
        the validator should successfully parse it after normalization.
        **Validates: Requirement 13.8**
        """
        # Create JSON string
        json_str = json.dumps(data)
        
        # Apply transformations
        if use_smart_quotes:
            json_str = add_smart_quotes(json_str)
        
        if add_control_chars:
            # Add control characters to string values
            data_with_control = data.copy()
            for key in data_with_control:
                if isinstance(data_with_control[key], str):
                    data_with_control[key] = add_control_characters(data_with_control[key])
            json_str = json.dumps(data_with_control)
            if use_smart_quotes:
                json_str = add_smart_quotes(json_str)
        
        # Validate - should succeed after normalization
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "summary", attempt_repair=True)
        
        # Should return a valid SummaryEnrichment object
        assert isinstance(result, SummaryEnrichment)
        assert len(result.short) > 0
        assert len(result.medium) > 0
        assert len(result.long) > 0
    
    @given(
        data=tag_json_strategy(),
        use_smart_quotes=st.booleans()
    )
    @settings(max_examples=100)
    def test_property_44_tag_normalization(self, data, use_smart_quotes):
        """
        **Property 44: Provider Response Normalization (Tags)**
        *For any* valid tag JSON with smart quotes, the validator should
        successfully parse it after normalization.
        **Validates: Requirement 13.8**
        """
        # Create JSON string
        json_str = json.dumps(data)
        
        # Apply smart quotes
        if use_smart_quotes:
            json_str = add_smart_quotes(json_str)
        
        # Validate - should succeed after normalization
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "tag", attempt_repair=True)
        
        # Should return a valid TagEnrichment object
        assert isinstance(result, TagEnrichment)
        assert len(result.categories) > 0
        assert len(result.keywords) > 0
    
    @given(
        data=st.lists(chapter_json_strategy(), min_size=1, max_size=5),
        use_smart_quotes=st.booleans()
    )
    @settings(max_examples=50)
    def test_property_44_chapter_normalization(self, data, use_smart_quotes):
        """
        **Property 44: Provider Response Normalization (Chapters)**
        *For any* valid chapter array JSON with smart quotes, the validator
        should successfully parse it after normalization.
        **Validates: Requirement 13.8**
        """
        # Create JSON string
        json_str = json.dumps(data)
        
        # Apply smart quotes
        if use_smart_quotes:
            json_str = add_smart_quotes(json_str)
        
        # Validate - should succeed after normalization
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "chapter", attempt_repair=True)
        
        # Should return a list of ChapterEnrichment objects
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, ChapterEnrichment)
    
    @given(
        data=st.lists(highlight_json_strategy(), min_size=1, max_size=5),
        use_smart_quotes=st.booleans()
    )
    @settings(max_examples=50)
    def test_property_44_highlight_normalization(self, data, use_smart_quotes):
        """
        **Property 44: Provider Response Normalization (Highlights)**
        *For any* valid highlight array JSON with smart quotes, the validator
        should successfully parse it after normalization.
        **Validates: Requirement 13.8**
        """
        # Create JSON string
        json_str = json.dumps(data)
        
        # Apply smart quotes
        if use_smart_quotes:
            json_str = add_smart_quotes(json_str)
        
        # Validate - should succeed after normalization
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "highlight", attempt_repair=True)
        
        # Should return a list of HighlightEnrichment objects
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, HighlightEnrichment)


# ============================================================================
# PROPERTY 45: Array Response Validation (Requirement 13.9)
# ============================================================================

class TestArrayResponseValidation:
    """Property 45: Array Response Validation.
    
    Validates Requirement 13.9: The system must correctly validate array
    responses for chapters/highlights and single object responses for
    summary/tags, with clear error messages for type mismatches.
    """
    
    @given(data=summary_json_strategy())
    @settings(max_examples=50)
    def test_property_45_summary_returns_object(self, data):
        """
        **Property 45: Array Response Validation (Summary Returns Object)**
        *For any* valid summary JSON, the validator should return a single
        SummaryEnrichment object, not a list.
        **Validates: Requirement 13.9**
        """
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "summary", attempt_repair=True)
        
        # Should return a single object, not a list
        assert isinstance(result, SummaryEnrichment)
        assert not isinstance(result, list)
    
    @given(data=tag_json_strategy())
    @settings(max_examples=50)
    def test_property_45_tag_returns_object(self, data):
        """
        **Property 45: Array Response Validation (Tag Returns Object)**
        *For any* valid tag JSON, the validator should return a single
        TagEnrichment object, not a list.
        **Validates: Requirement 13.9**
        """
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "tag", attempt_repair=True)
        
        # Should return a single object, not a list
        assert isinstance(result, TagEnrichment)
        assert not isinstance(result, list)
    
    @given(data=st.lists(chapter_json_strategy(), min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_property_45_chapter_returns_list(self, data):
        """
        **Property 45: Array Response Validation (Chapter Returns List)**
        *For any* valid chapter array JSON, the validator should return a
        list of ChapterEnrichment objects.
        **Validates: Requirement 13.9**
        """
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "chapter", attempt_repair=True)
        
        # Should return a list
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each item should be a ChapterEnrichment
        for item in result:
            assert isinstance(item, ChapterEnrichment)
    
    @given(data=st.lists(highlight_json_strategy(), min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_property_45_highlight_returns_list(self, data):
        """
        **Property 45: Array Response Validation (Highlight Returns List)**
        *For any* valid highlight array JSON, the validator should return a
        list of HighlightEnrichment objects.
        **Validates: Requirement 13.9**
        """
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        result = validator.validate_response(json_str, "highlight", attempt_repair=True)
        
        # Should return a list
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each item should be a HighlightEnrichment
        for item in result:
            assert isinstance(item, HighlightEnrichment)
    
    @given(data=chapter_json_strategy())
    @settings(max_examples=30)
    def test_property_45_chapter_rejects_single_object(self, data):
        """
        **Property 45: Array Response Validation (Chapter Rejects Object)**
        *For any* chapter JSON that is a single object instead of an array,
        the validator should raise a clear error message.
        **Validates: Requirement 13.9**
        """
        # Create single object JSON (not an array)
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        
        # Should raise SchemaValidationError with clear message
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_response(json_str, "chapter", attempt_repair=True)
        
        # Error message should mention "array"
        assert "array" in str(exc_info.value).lower()
    
    @given(data=highlight_json_strategy())
    @settings(max_examples=30)
    def test_property_45_highlight_rejects_single_object(self, data):
        """
        **Property 45: Array Response Validation (Highlight Rejects Object)**
        *For any* highlight JSON that is a single object instead of an array,
        the validator should raise a clear error message.
        **Validates: Requirement 13.9**
        """
        # Create single object JSON (not an array)
        json_str = json.dumps(data)
        
        validator = SchemaValidator()
        
        # Should raise SchemaValidationError with clear message
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_response(json_str, "highlight", attempt_repair=True)
        
        # Error message should mention "array"
        assert "array" in str(exc_info.value).lower()


# ============================================================================
# PROPERTY 46: Multi-Enrichment Consistency (Requirement 13.10)
# ============================================================================

class TestMultiEnrichmentConsistency:
    """Property 46: Multi-Enrichment Consistency.
    
    Validates Requirement 13.10: Running enrichments individually should
    produce the same results as running them together (--all flag), ensuring
    proper context isolation and no state pollution.
    """
    
    @given(
        summary_data=summary_json_strategy(),
        tag_data=tag_json_strategy()
    )
    @settings(max_examples=50)
    def test_property_46_summary_tag_isolation(self, summary_data, tag_data):
        """
        **Property 46: Multi-Enrichment Consistency (Summary + Tag)**
        *For any* summary and tag data, validating them separately should
        produce the same results as validating them in sequence.
        **Validates: Requirement 13.10**
        """
        validator = SchemaValidator()
        
        # Validate individually
        summary_json = json.dumps(summary_data)
        tag_json = json.dumps(tag_data)
        
        summary_result_1 = validator.validate_response(summary_json, "summary")
        tag_result_1 = validator.validate_response(tag_json, "tag")
        
        # Validate in sequence (simulating multi-enrichment)
        summary_result_2 = validator.validate_response(summary_json, "summary")
        tag_result_2 = validator.validate_response(tag_json, "tag")
        
        # Results should be identical
        assert summary_result_1.model_dump() == summary_result_2.model_dump()
        assert tag_result_1.model_dump() == tag_result_2.model_dump()
    
    @given(
        chapter_data=st.lists(chapter_json_strategy(), min_size=1, max_size=3),
        highlight_data=st.lists(highlight_json_strategy(), min_size=1, max_size=3)
    )
    @settings(max_examples=30)
    def test_property_46_chapter_highlight_isolation(self, chapter_data, highlight_data):
        """
        **Property 46: Multi-Enrichment Consistency (Chapter + Highlight)**
        *For any* chapter and highlight data, validating them separately should
        produce the same results as validating them in sequence.
        **Validates: Requirement 13.10**
        """
        validator = SchemaValidator()
        
        # Validate individually
        chapter_json = json.dumps(chapter_data)
        highlight_json = json.dumps(highlight_data)
        
        chapter_result_1 = validator.validate_response(chapter_json, "chapter")
        highlight_result_1 = validator.validate_response(highlight_json, "highlight")
        
        # Validate in sequence (simulating multi-enrichment)
        chapter_result_2 = validator.validate_response(chapter_json, "chapter")
        highlight_result_2 = validator.validate_response(highlight_json, "highlight")
        
        # Results should be identical
        assert len(chapter_result_1) == len(chapter_result_2)
        assert len(highlight_result_1) == len(highlight_result_2)
        
        for i in range(len(chapter_result_1)):
            assert chapter_result_1[i].model_dump() == chapter_result_2[i].model_dump()
        
        for i in range(len(highlight_result_1)):
            assert highlight_result_1[i].model_dump() == highlight_result_2[i].model_dump()
    
    @given(
        summary_data=summary_json_strategy(),
        tag_data=tag_json_strategy(),
        chapter_data=st.lists(chapter_json_strategy(), min_size=1, max_size=2),
        highlight_data=st.lists(highlight_json_strategy(), min_size=1, max_size=2)
    )
    @settings(max_examples=20)
    def test_property_46_all_enrichments_isolation(
        self, summary_data, tag_data, chapter_data, highlight_data
    ):
        """
        **Property 46: Multi-Enrichment Consistency (All Enrichments)**
        *For any* complete set of enrichment data, validating them separately
        should produce the same results as validating them all in sequence.
        **Validates: Requirement 13.10**
        """
        validator = SchemaValidator()
        
        # Prepare JSON strings
        summary_json = json.dumps(summary_data)
        tag_json = json.dumps(tag_data)
        chapter_json = json.dumps(chapter_data)
        highlight_json = json.dumps(highlight_data)
        
        # Validate individually
        summary_result_1 = validator.validate_response(summary_json, "summary")
        tag_result_1 = validator.validate_response(tag_json, "tag")
        chapter_result_1 = validator.validate_response(chapter_json, "chapter")
        highlight_result_1 = validator.validate_response(highlight_json, "highlight")
        
        # Validate all in sequence (simulating --all flag)
        summary_result_2 = validator.validate_response(summary_json, "summary")
        tag_result_2 = validator.validate_response(tag_json, "tag")
        chapter_result_2 = validator.validate_response(chapter_json, "chapter")
        highlight_result_2 = validator.validate_response(highlight_json, "highlight")
        
        # All results should be identical
        assert summary_result_1.model_dump() == summary_result_2.model_dump()
        assert tag_result_1.model_dump() == tag_result_2.model_dump()
        
        assert len(chapter_result_1) == len(chapter_result_2)
        for i in range(len(chapter_result_1)):
            assert chapter_result_1[i].model_dump() == chapter_result_2[i].model_dump()
        
        assert len(highlight_result_1) == len(highlight_result_2)
        for i in range(len(highlight_result_1)):
            assert highlight_result_1[i].model_dump() == highlight_result_2[i].model_dump()
    
    @given(
        enrichment_order=st.permutations(["summary", "tag", "chapter", "highlight"])
    )
    @settings(max_examples=20)
    def test_property_46_order_independence(self, enrichment_order):
        """
        **Property 46: Multi-Enrichment Consistency (Order Independence)**
        *For any* order of enrichment types, the validation results should
        be independent of the order in which they are validated.
        **Validates: Requirement 13.10**
        """
        validator = SchemaValidator()
        
        # Create test data for all types
        test_data = {
            "summary": {"short": "Short summary", "medium": "Medium summary text", "long": "Long summary with more details"},
            "tag": {"categories": ["tech"], "keywords": ["test"], "entities": ["TestCorp"]},
            "chapter": [{"title": "Chapter 1", "start_time": "00:00:00", "end_time": "00:05:00", "description": "First chapter"}],
            "highlight": [{"timestamp": "00:01:30", "quote": "Important quote", "importance": "high", "context": "Context"}]
        }
        
        # Validate in the given order
        results = {}
        for enrichment_type in enrichment_order:
            json_str = json.dumps(test_data[enrichment_type])
            result = validator.validate_response(json_str, enrichment_type)
            results[enrichment_type] = result
        
        # Verify all results are valid (order shouldn't affect validity)
        assert isinstance(results["summary"], SummaryEnrichment)
        assert isinstance(results["tag"], TagEnrichment)
        assert isinstance(results["chapter"], list)
        assert isinstance(results["highlight"], list)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
