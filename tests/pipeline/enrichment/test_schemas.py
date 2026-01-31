"""
Unit Tests for Enrichment Schemas

Tests all enrichment schema models (EnrichmentV1, Summary, Tag, Chapter, Highlight)
to ensure proper validation, serialization, and error handling.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from pipeline.enrichment.schemas.enrichment_v1 import (
    EnrichmentV1,
    EnrichmentMetadata
)
from pipeline.enrichment.schemas.summary import SummaryEnrichment
from pipeline.enrichment.schemas.tag import TagEnrichment
from pipeline.enrichment.schemas.chapter import ChapterEnrichment
from pipeline.enrichment.schemas.highlight import (
    HighlightEnrichment,
    ImportanceLevel
)
from tests.fixtures.mock_llm_responses import (
    MOCK_SUMMARY_RESPONSE,
    MOCK_TAG_RESPONSE,
    MOCK_CHAPTERS_RESPONSE,
    MOCK_HIGHLIGHTS_RESPONSE,
    MOCK_COMPLETE_ENRICHMENT
)


class TestSummaryEnrichment:
    """Test SummaryEnrichment schema."""
    
    def test_valid_summary(self):
        """Test creating valid summary enrichment."""
        summary = SummaryEnrichment(**MOCK_SUMMARY_RESPONSE)
        assert summary.short == MOCK_SUMMARY_RESPONSE["short"]
        assert summary.medium == MOCK_SUMMARY_RESPONSE["medium"]
        assert summary.long == MOCK_SUMMARY_RESPONSE["long"]
    
    def test_summary_max_lengths(self):
        """Test summary length constraints."""
        # Short summary too long (>500 chars)
        with pytest.raises(ValidationError):
            SummaryEnrichment(
                short="x" * 501,
                medium="Valid medium",
                long="Valid long"
            )
        
        # Medium summary too long (>2000 chars)
        with pytest.raises(ValidationError):
            SummaryEnrichment(
                short="Valid short",
                medium="x" * 2001,
                long="Valid long"
            )
        
        # Long summary too long (>5000 chars)
        with pytest.raises(ValidationError):
            SummaryEnrichment(
                short="Valid short",
                medium="Valid medium",
                long="x" * 5001
            )
    
    def test_summary_missing_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            SummaryEnrichment(short="Only short")
    
    def test_summary_serialization(self):
        """Test JSON serialization."""
        summary = SummaryEnrichment(**MOCK_SUMMARY_RESPONSE)
        json_data = summary.model_dump()
        assert json_data["short"] == MOCK_SUMMARY_RESPONSE["short"]
        
        # Test round-trip
        summary2 = SummaryEnrichment(**json_data)
        assert summary == summary2


class TestTagEnrichment:
    """Test TagEnrichment schema."""
    
    def test_valid_tags(self):
        """Test creating valid tag enrichment."""
        tags = TagEnrichment(**MOCK_TAG_RESPONSE)
        assert len(tags.categories) > 0
        assert len(tags.keywords) > 0
        assert len(tags.entities) >= 0  # Entities can be empty
    
    def test_tags_min_items(self):
        """Test minimum item requirements."""
        # Categories must have at least 1 item
        with pytest.raises(ValidationError):
            TagEnrichment(
                categories=[],
                keywords=["test"],
                entities=[]
            )
        
        # Keywords must have at least 1 item
        with pytest.raises(ValidationError):
            TagEnrichment(
                categories=["test"],
                keywords=[],
                entities=[]
            )
    
    def test_tags_entities_optional(self):
        """Test that entities can be empty."""
        tags = TagEnrichment(
            categories=["Technology"],
            keywords=["AI", "ML"],
            entities=[]
        )
        assert tags.entities == []
    
    def test_tags_serialization(self):
        """Test JSON serialization."""
        tags = TagEnrichment(**MOCK_TAG_RESPONSE)
        json_data = tags.model_dump()
        assert json_data["categories"] == MOCK_TAG_RESPONSE["categories"]


class TestChapterEnrichment:
    """Test ChapterEnrichment schema."""
    
    def test_valid_chapter(self):
        """Test creating valid chapter enrichment."""
        chapter = ChapterEnrichment(**MOCK_CHAPTERS_RESPONSE[0])
        assert chapter.title == MOCK_CHAPTERS_RESPONSE[0]["title"]
        assert chapter.start_time == MOCK_CHAPTERS_RESPONSE[0]["start_time"]
        assert chapter.end_time == MOCK_CHAPTERS_RESPONSE[0]["end_time"]
    
    def test_chapter_timestamp_format(self):
        """Test timestamp format validation."""
        # Invalid format
        with pytest.raises(ValidationError):
            ChapterEnrichment(
                title="Test",
                start_time="5:30",  # Should be 00:05:30
                end_time="00:10:00",
                description="Test"
            )
        
        # Invalid format
        with pytest.raises(ValidationError):
            ChapterEnrichment(
                title="Test",
                start_time="00:05:30",
                end_time="10:00",  # Should be 00:10:00
                description="Test"
            )
    
    def test_chapter_end_after_start(self):
        """Test that end_time must be after start_time."""
        with pytest.raises(ValidationError):
            ChapterEnrichment(
                title="Test",
                start_time="00:10:00",
                end_time="00:05:00",  # Before start
                description="Test"
            )
    
    def test_chapter_title_length(self):
        """Test title length constraints."""
        # Title too long (>200 chars)
        with pytest.raises(ValidationError):
            ChapterEnrichment(
                title="x" * 201,
                start_time="00:00:00",
                end_time="00:10:00",
                description="Test"
            )
    
    def test_chapter_description_length(self):
        """Test description length constraints."""
        # Description too long (>500 chars)
        with pytest.raises(ValidationError):
            ChapterEnrichment(
                title="Test",
                start_time="00:00:00",
                end_time="00:10:00",
                description="x" * 501
            )


class TestHighlightEnrichment:
    """Test HighlightEnrichment schema."""
    
    def test_valid_highlight(self):
        """Test creating valid highlight enrichment."""
        highlight = HighlightEnrichment(**MOCK_HIGHLIGHTS_RESPONSE[0])
        assert highlight.timestamp == MOCK_HIGHLIGHTS_RESPONSE[0]["timestamp"]
        assert highlight.quote == MOCK_HIGHLIGHTS_RESPONSE[0]["quote"]
        assert highlight.importance == ImportanceLevel.HIGH
    
    def test_highlight_importance_levels(self):
        """Test all importance levels."""
        for level in [ImportanceLevel.HIGH, ImportanceLevel.MEDIUM, ImportanceLevel.LOW]:
            highlight = HighlightEnrichment(
                timestamp="00:05:00",
                quote="Test quote",
                importance=level
            )
            assert highlight.importance == level
    
    def test_highlight_timestamp_format(self):
        """Test timestamp format validation."""
        with pytest.raises(ValidationError):
            HighlightEnrichment(
                timestamp="5:30",  # Should be 00:05:30
                quote="Test",
                importance=ImportanceLevel.HIGH
            )
    
    def test_highlight_quote_length(self):
        """Test quote length constraints."""
        # Quote too long (>1000 chars)
        with pytest.raises(ValidationError):
            HighlightEnrichment(
                timestamp="00:05:00",
                quote="x" * 1001,
                importance=ImportanceLevel.HIGH
            )
    
    def test_highlight_context_optional(self):
        """Test that context is optional."""
        highlight = HighlightEnrichment(
            timestamp="00:05:00",
            quote="Test quote",
            importance=ImportanceLevel.HIGH
        )
        assert highlight.context is None
    
    def test_highlight_context_length(self):
        """Test context length constraints."""
        # Context too long (>500 chars)
        with pytest.raises(ValidationError):
            HighlightEnrichment(
                timestamp="00:05:00",
                quote="Test",
                importance=ImportanceLevel.HIGH,
                context="x" * 501
            )


class TestEnrichmentMetadata:
    """Test EnrichmentMetadata schema."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = EnrichmentMetadata(
            provider="openai",
            model="gpt-4-turbo",
            cost_usd=0.31,
            tokens_used=8300,
            enrichment_types=["summary", "tags"]
        )
        assert metadata.provider == "openai"
        assert metadata.cost_usd == 0.31
        assert metadata.cache_hit is False  # Default
    
    def test_metadata_cost_non_negative(self):
        """Test that cost must be non-negative."""
        with pytest.raises(ValidationError):
            EnrichmentMetadata(
                provider="openai",
                model="gpt-4",
                cost_usd=-0.10,  # Negative cost
                tokens_used=1000,
                enrichment_types=["summary"]
            )
    
    def test_metadata_tokens_non_negative(self):
        """Test that tokens must be non-negative."""
        with pytest.raises(ValidationError):
            EnrichmentMetadata(
                provider="openai",
                model="gpt-4",
                cost_usd=0.10,
                tokens_used=-100,  # Negative tokens
                enrichment_types=["summary"]
            )
    
    def test_metadata_enrichment_types_required(self):
        """Test that at least one enrichment type is required."""
        with pytest.raises(ValidationError):
            EnrichmentMetadata(
                provider="openai",
                model="gpt-4",
                cost_usd=0.10,
                tokens_used=1000,
                enrichment_types=[]  # Empty list
            )


class TestEnrichmentV1:
    """Test EnrichmentV1 container schema."""
    
    def test_valid_complete_enrichment(self):
        """Test creating complete enrichment with all types."""
        enrichment = EnrichmentV1(**MOCK_COMPLETE_ENRICHMENT)
        assert enrichment.enrichment_version == "v1"
        assert enrichment.summary is not None
        assert enrichment.tags is not None
        assert enrichment.chapters is not None
        assert enrichment.highlights is not None
    
    def test_enrichment_with_single_type(self):
        """Test enrichment with only one type."""
        enrichment = EnrichmentV1(
            metadata=EnrichmentMetadata(
                provider="openai",
                model="gpt-4",
                cost_usd=0.10,
                tokens_used=1000,
                enrichment_types=["summary"]
            ),
            summary=SummaryEnrichment(**MOCK_SUMMARY_RESPONSE)
        )
        assert enrichment.summary is not None
        assert enrichment.tags is None
    
    def test_enrichment_requires_at_least_one_type(self):
        """Test that at least one enrichment type is required."""
        with pytest.raises(ValueError, match="At least one enrichment type"):
            EnrichmentV1(
                metadata=EnrichmentMetadata(
                    provider="openai",
                    model="gpt-4",
                    cost_usd=0.10,
                    tokens_used=1000,
                    enrichment_types=["summary"]
                )
                # No enrichment types provided
            )
    
    def test_enrichment_serialization(self):
        """Test JSON serialization of complete enrichment."""
        enrichment = EnrichmentV1(**MOCK_COMPLETE_ENRICHMENT)
        json_data = enrichment.model_dump()
        
        assert json_data["enrichment_version"] == "v1"
        assert "metadata" in json_data
        assert "summary" in json_data
        
        # Test round-trip
        enrichment2 = EnrichmentV1(**json_data)
        assert enrichment.enrichment_version == enrichment2.enrichment_version
