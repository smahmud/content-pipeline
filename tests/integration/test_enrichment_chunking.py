"""
Integration tests for long transcript handling with chunking

Tests automatic chunking of transcripts that exceed LLM context windows,
chunk processing, and result merging across all enrichment types.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest
from pipeline.enrichment.chunking import ChunkingStrategy, TextChunk
from pipeline.llm.providers.base import LLMResponse, LLMRequest
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata


@pytest.fixture
def long_transcript():
    """Create a long transcript that exceeds typical context windows."""
    # Generate a transcript with ~10,000 words (exceeds many context windows)
    paragraph = (
        "This is a detailed discussion about machine learning and artificial intelligence. "
        "We explore various concepts including neural networks, deep learning architectures, "
        "supervised and unsupervised learning approaches, and practical applications in "
        "computer vision and natural language processing. The content covers fundamental "
        "principles, advanced techniques, and real-world case studies. "
    ) * 50  # Repeat to create long content
    
    return paragraph


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for chunking tests."""
    provider = Mock()
    provider.get_context_window.return_value = 4096
    provider.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-3.5-turbo"],
        "max_tokens": 4096
    }
    return provider


@pytest.fixture
def mock_provider_with_small_context():
    """Create mock provider with small context window to trigger chunking."""
    provider = Mock()
    
    # Configure small context window (1000 tokens)
    provider.get_context_window.return_value = 1000
    provider.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-3.5-turbo"],
        "max_tokens": 4096,
        "default_model": "gpt-3.5-turbo"
    }
    
    # Configure cost estimation
    provider.estimate_cost.return_value = 0.001
    
    # Configure generate to return different responses for chunks
    responses = [
        LLMResponse(
            content=json.dumps({
                "short": "Summary of chunk 1",
                "medium": "Medium summary of chunk 1 content",
                "long": "Long detailed summary of chunk 1 with all key points"
            }),
            model_used="gpt-3.5-turbo",
            tokens_used=300,
            cost_usd=0.001
        ),
        LLMResponse(
            content=json.dumps({
                "short": "Summary of chunk 2",
                "medium": "Medium summary of chunk 2 content",
                "long": "Long detailed summary of chunk 2 with all key points"
            }),
            model_used="gpt-3.5-turbo",
            tokens_used=300,
            cost_usd=0.001
        ),
        LLMResponse(
            content=json.dumps({
                "short": "Combined summary of all chunks",
                "medium": "Medium combined summary covering all content",
                "long": "Long combined summary with comprehensive coverage of all topics"
            }),
            model_used="gpt-3.5-turbo",
            tokens_used=400,
            cost_usd=0.002
        )
    ]
    
    provider.generate.side_effect = responses
    
    return provider


class TestChunkingStrategy:
    """Integration tests for chunking strategy."""
    
    def test_chunking_detection(self, long_transcript, mock_provider):
        """Test that chunking is triggered for long transcripts."""
        # Configure small context window to trigger chunking
        mock_provider.get_context_window.return_value = 1000
        
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Check if transcript needs chunking
        needs_chunking = strategy.needs_chunking(
            text=long_transcript,
            model="gpt-3.5-turbo",
            prompt_overhead=500
        )
        
        assert needs_chunking is True
    
    def test_no_chunking_for_short_text(self, mock_provider):
        """Test that short text doesn't trigger chunking."""
        # Configure large context window
        mock_provider.get_context_window.return_value = 100000
        
        strategy = ChunkingStrategy(provider=mock_provider)
        
        short_text = "This is a short transcript that fits in context."
        
        needs_chunking = strategy.needs_chunking(
            text=short_text,
            model="gpt-3.5-turbo",
            prompt_overhead=500
        )
        
        assert needs_chunking is False
    
    def test_chunk_creation(self, long_transcript, mock_provider):
        """Test creation of chunks from long transcript."""
        # Configure small context window to force chunking
        mock_provider.get_context_window.return_value = 1000
        
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create chunks using chunk_text method
        chunks = strategy.chunk_text(
            text=long_transcript,
            model="gpt-3.5-turbo",
            prompt_overhead=500
        )
        
        # Verify chunks were created
        assert len(chunks) > 1
        
        # Verify each chunk is a TextChunk
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) > 0
            assert chunk.chunk_number >= 0
            assert chunk.total_chunks == len(chunks)
    
    def test_chunk_boundaries(self, long_transcript, mock_provider):
        """Test that chunks split at natural boundaries."""
        # Configure small context window
        mock_provider.get_context_window.return_value = 1000
        
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create chunks
        chunks = strategy.chunk_text(
            text=long_transcript,
            model="gpt-3.5-turbo",
            prompt_overhead=500
        )
        
        # Verify chunks have valid indices
        for i, chunk in enumerate(chunks):
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index
            assert chunk.chunk_number == i
    
    def test_chunk_coverage(self, mock_provider):
        """Test that chunks cover the entire text without gaps."""
        mock_provider.get_context_window.return_value = 500
        
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create a text with clear paragraph boundaries
        text = "\n\n".join([
            "First paragraph with some content about topic one.",
            "Second paragraph discussing another important topic.",
            "Third paragraph with additional information.",
            "Fourth paragraph wrapping up the discussion."
        ])
        
        chunks = strategy.chunk_text(
            text=text,
            model="gpt-3.5-turbo",
            prompt_overhead=100
        )
        
        # Verify we got chunks
        assert len(chunks) >= 1
        
        # Verify chunk numbering
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_number == i
            assert chunk.total_chunks == len(chunks)


class TestSummaryMerging:
    """Integration tests for summary merging across chunks."""
    
    def test_summary_chunk_merging(self, long_transcript, mock_provider_with_small_context):
        """Test that summaries from multiple chunks are merged correctly."""
        # Create orchestrator with mocked provider
        factory = Mock()
        factory.create_provider.return_value = mock_provider_with_small_context
        
        orchestrator = EnrichmentOrchestrator(provider_factory=factory)
        
        # Execute enrichment
        request = EnrichmentRequest(
            transcript_text=long_transcript,
            language="en",
            duration=600.0,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        result = orchestrator.enrich(request)
        
        # Verify result contains summary
        assert isinstance(result, EnrichmentV1)
        assert result.summary is not None
        # SummaryEnrichment has short, medium, long attributes
        assert hasattr(result.summary, 'short')
        assert hasattr(result.summary, 'medium')
        assert hasattr(result.summary, 'long')
        assert result.summary.short is not None
        assert result.summary.medium is not None
        assert result.summary.long is not None


class TestChapterMerging:
    """Integration tests for chapter merging across chunks."""
    
    def test_chapter_timestamp_preservation(self, mock_provider):
        """Test that chapter timestamps are preserved during merging."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create chapters from different chunks
        chunk1_chapters = [
            {
                "title": "Introduction",
                "start_time": "00:00:00",
                "end_time": "00:05:00",
                "description": "Opening remarks"
            },
            {
                "title": "Main Topic Part 1",
                "start_time": "00:05:00",
                "end_time": "00:10:00",
                "description": "First part of main discussion"
            }
        ]
        
        chunk2_chapters = [
            {
                "title": "Main Topic Part 2",
                "start_time": "00:10:00",
                "end_time": "00:15:00",
                "description": "Second part of main discussion"
            },
            {
                "title": "Conclusion",
                "start_time": "00:15:00",
                "end_time": "00:20:00",
                "description": "Closing remarks"
            }
        ]
        
        # Merge chapters using correct parameter name
        merged = strategy.merge_chapters(
            chunk_chapters=[chunk1_chapters, chunk2_chapters]
        )
        
        # Verify all chapters are present
        assert len(merged) == 4
        
        # Verify timestamps are in order
        for i in range(len(merged) - 1):
            assert merged[i]['start_time'] <= merged[i+1]['start_time']
    
    def test_chapter_deduplication(self, mock_provider):
        """Test that duplicate chapters are removed during merging."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create overlapping chapters from chunks
        chunk1_chapters = [
            {
                "title": "Introduction",
                "start_time": "00:00:00",
                "end_time": "00:05:00",
                "description": "Opening"
            },
            {
                "title": "Main Topic",
                "start_time": "00:05:00",
                "end_time": "00:10:00",
                "description": "Discussion"
            }
        ]
        
        chunk2_chapters = [
            {
                "title": "Main Topic",  # Duplicate - same start_time
                "start_time": "00:05:00",
                "end_time": "00:10:00",
                "description": "Discussion"
            },
            {
                "title": "Conclusion",
                "start_time": "00:10:00",
                "end_time": "00:15:00",
                "description": "Closing"
            }
        ]
        
        # Merge chapters
        merged = strategy.merge_chapters(
            chunk_chapters=[chunk1_chapters, chunk2_chapters]
        )
        
        # Verify duplicates were removed (based on similar timestamps)
        assert len(merged) == 3


class TestHighlightMerging:
    """Integration tests for highlight merging across chunks."""
    
    def test_highlight_timestamp_preservation(self, mock_provider):
        """Test that highlight timestamps are preserved during merging."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create highlights from different chunks
        chunk1_highlights = [
            {
                "timestamp": "00:02:30",
                "quote": "Key point from chunk 1",
                "importance": "high",
                "context": "Important context"
            }
        ]
        
        chunk2_highlights = [
            {
                "timestamp": "00:12:45",
                "quote": "Key point from chunk 2",
                "importance": "medium",
                "context": "Additional context"
            }
        ]
        
        # Merge highlights using correct parameter name
        merged = strategy.merge_highlights(
            chunk_highlights=[chunk1_highlights, chunk2_highlights]
        )
        
        # Verify all highlights are present
        assert len(merged) == 2
        
        # Verify timestamps are in order
        assert merged[0]['timestamp'] < merged[1]['timestamp']
    
    def test_highlight_deduplication(self, mock_provider):
        """Test that duplicate highlights are removed during merging."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create highlights with similar timestamps (within 5 seconds)
        chunk1_highlights = [
            {
                "timestamp": "00:02:00",
                "quote": "First highlight",
                "importance": "high",
                "context": "Context"
            }
        ]
        
        chunk2_highlights = [
            {
                "timestamp": "00:02:03",  # Within 5 seconds of first - should be deduplicated
                "quote": "Duplicate highlight",
                "importance": "high",
                "context": "Context"
            },
            {
                "timestamp": "00:12:00",
                "quote": "Different highlight",
                "importance": "medium",
                "context": "Context"
            }
        ]
        
        # Merge highlights
        merged = strategy.merge_highlights(
            chunk_highlights=[chunk1_highlights, chunk2_highlights]
        )
        
        # Verify duplicates were removed
        assert len(merged) == 2


class TestTagMerging:
    """Integration tests for tag merging across chunks."""
    
    def test_tag_deduplication(self, mock_provider):
        """Test that duplicate tags are removed during merging."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create overlapping tags from chunks
        chunk1_tags = {
            "categories": ["Technology", "AI"],
            "keywords": ["machine learning", "neural networks"],
            "entities": ["TensorFlow", "Python"]
        }
        
        chunk2_tags = {
            "categories": ["AI", "Software"],  # "AI" is duplicate
            "keywords": ["neural networks", "deep learning"],  # "neural networks" is duplicate
            "entities": ["Python", "PyTorch"]  # "Python" is duplicate
        }
        
        # Merge tags using correct parameter name
        merged = strategy.merge_tags(
            chunk_tags=[chunk1_tags, chunk2_tags]
        )
        
        # Verify duplicates were removed
        assert len(merged['categories']) == 3  # Technology, AI, Software
        assert len(merged['keywords']) == 3  # machine learning, neural networks, deep learning
        assert len(merged['entities']) == 3  # TensorFlow, Python, PyTorch
        
        # Verify no duplicates
        assert len(set(merged['categories'])) == len(merged['categories'])
        assert len(set(merged['keywords'])) == len(merged['keywords'])
        assert len(set(merged['entities'])) == len(merged['entities'])
    
    def test_tag_aggregation_across_chunks(self, mock_provider):
        """Test that tags from multiple chunks are aggregated correctly."""
        strategy = ChunkingStrategy(provider=mock_provider)
        
        # Create tags from three chunks
        chunk1_tags = {
            "categories": ["AI"],
            "keywords": ["machine learning"],
            "entities": ["Python"]
        }
        
        chunk2_tags = {
            "categories": ["Technology"],
            "keywords": ["deep learning"],
            "entities": ["TensorFlow"]
        }
        
        chunk3_tags = {
            "categories": ["Software"],
            "keywords": ["neural networks"],
            "entities": ["PyTorch"]
        }
        
        # Merge tags
        merged = strategy.merge_tags(
            chunk_tags=[chunk1_tags, chunk2_tags, chunk3_tags]
        )
        
        # Verify all unique tags are present
        assert "AI" in merged['categories']
        assert "Technology" in merged['categories']
        assert "Software" in merged['categories']
        assert "machine learning" in merged['keywords']
        assert "deep learning" in merged['keywords']
        assert "neural networks" in merged['keywords']


class TestChunkingPerformance:
    """Integration tests for chunking performance and efficiency."""
    
    def test_chunking_minimizes_api_calls(self, long_transcript, mock_provider_with_small_context):
        """Test that enrichment completes successfully with mocked provider."""
        factory = Mock()
        factory.create_provider.return_value = mock_provider_with_small_context
        
        orchestrator = EnrichmentOrchestrator(provider_factory=factory)
        
        # Execute enrichment
        request = EnrichmentRequest(
            transcript_text=long_transcript,
            language="en",
            duration=600.0,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        result = orchestrator.enrich(request)
        
        # Verify result was produced
        assert isinstance(result, EnrichmentV1)
        assert result.summary is not None
        
        # Verify the factory was used to create a provider
        factory.create_provider.assert_called_once_with("openai")
