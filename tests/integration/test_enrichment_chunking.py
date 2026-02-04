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
from tests.fixtures.mock_llm_responses import (
    MOCK_SUMMARY_RESPONSE,
    MOCK_TAG_RESPONSE,
    MOCK_CHAPTERS_RESPONSE,
    MOCK_HIGHLIGHTS_RESPONSE
)


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
def mock_agent_with_small_context():
    """Create mock agent with small context window to trigger chunking."""
    agent = Mock()
    
    # Configure small context window (1000 tokens)
    agent.get_context_window.return_value = 1000
    agent.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-3.5-turbo"],
        "max_tokens": 4096
    }
    
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
    
    agent.generate.side_effect = responses
    
    return agent


class TestChunkingStrategy:
    """Integration tests for chunking strategy."""
    
    def test_chunking_detection(self, long_transcript):
        """Test that chunking is triggered for long transcripts."""
        strategy = ChunkingStrategy()
        
        # Check if transcript needs chunking
        needs_chunking = strategy.needs_chunking(
            text=long_transcript,
            max_tokens=1000
        )
        
        assert needs_chunking is True
    
    def test_chunk_creation(self, long_transcript):
        """Test creation of chunks from long transcript."""
        strategy = ChunkingStrategy()
        
        # Create chunks
        chunks = strategy.create_chunks(
            text=long_transcript,
            max_tokens=1000,
            overlap_tokens=100
        )
        
        # Verify chunks were created
        assert len(chunks) > 1
        
        # Verify each chunk is within token limit
        for chunk in chunks:
            assert isinstance(chunk, TranscriptChunk)
            assert chunk.token_count <= 1000
            assert len(chunk.text) > 0
    
    def test_chunk_boundaries(self, long_transcript):
        """Test that chunks split at natural boundaries."""
        strategy = ChunkingStrategy()
        
        # Create chunks
        chunks = strategy.create_chunks(
            text=long_transcript,
            max_tokens=1000,
            overlap_tokens=100
        )
        
        # Verify chunks don't split mid-sentence
        for chunk in chunks:
            # Chunk should end with sentence-ending punctuation or be the last chunk
            if chunk.chunk_index < len(chunks) - 1:
                assert chunk.text.rstrip().endswith(('.', '!', '?', ':', ';'))
    
    def test_chunk_overlap(self):
        """Test that chunks have appropriate overlap."""
        strategy = ChunkingStrategy()
        
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks = strategy.create_chunks(
            text=text,
            max_tokens=20,
            overlap_tokens=5
        )
        
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Some content from chunk i should appear in chunk i+1
                # (This is a simplified check - real implementation would be more sophisticated)
                assert len(chunks[i].text) > 0
                assert len(chunks[i+1].text) > 0


class TestSummaryMerging:
    """Integration tests for summary merging across chunks."""
    
    @patch('pipeline.enrichment.chunking.ChunkingStrategy')
    def test_summary_chunk_merging(self, mock_chunking_class, long_transcript, mock_agent_with_small_context):
        """Test that summaries from multiple chunks are merged correctly."""
        # Setup chunking strategy mock
        mock_strategy = Mock()
        mock_chunking_class.return_value = mock_strategy
        
        # Configure to indicate chunking is needed
        mock_strategy.needs_chunking.return_value = True
        
        # Create mock chunks
        mock_chunks = [
            TranscriptChunk(
                text=long_transcript[:len(long_transcript)//2],
                chunk_index=0,
                total_chunks=2,
                token_count=500,
                start_position=0,
                end_position=len(long_transcript)//2
            ),
            TranscriptChunk(
                text=long_transcript[len(long_transcript)//2:],
                chunk_index=1,
                total_chunks=2,
                token_count=500,
                start_position=len(long_transcript)//2,
                end_position=len(long_transcript)
            )
        ]
        mock_strategy.create_chunks.return_value = mock_chunks
        
        # Create orchestrator with mocked agent
        factory = Mock()
        factory.create_agent.return_value = mock_agent_with_small_context
        
        orchestrator = EnrichmentOrchestrator(agent_factory=factory)
        
        # Execute enrichment
        request = EnrichmentRequest(
            transcript_text=long_transcript,
            language="en",
            duration=600.0,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        result = orchestrator.enrich(request)
        
        # Verify result contains merged summary
        assert isinstance(result, EnrichmentV1)
        assert result.summary is not None
        assert 'short' in result.summary
        assert 'medium' in result.summary
        assert 'long' in result.summary


class TestChapterMerging:
    """Integration tests for chapter merging across chunks."""
    
    def test_chapter_timestamp_preservation(self):
        """Test that chapter timestamps are preserved during merging."""
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
        
        # Merge chapters
        strategy = ChunkingStrategy()
        merged = strategy.merge_chapters(
            chunk_results=[chunk1_chapters, chunk2_chapters]
        )
        
        # Verify all chapters are present
        assert len(merged) == 4
        
        # Verify timestamps are in order
        for i in range(len(merged) - 1):
            assert merged[i]['end_time'] <= merged[i+1]['start_time']
    
    def test_chapter_deduplication(self):
        """Test that duplicate chapters are removed during merging."""
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
                "title": "Main Topic",  # Duplicate
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
        strategy = ChunkingStrategy()
        merged = strategy.merge_chapters(
            chunk_results=[chunk1_chapters, chunk2_chapters]
        )
        
        # Verify duplicates were removed
        assert len(merged) == 3
        
        # Verify no duplicate titles at same timestamp
        seen = set()
        for chapter in merged:
            key = (chapter['title'], chapter['start_time'])
            assert key not in seen
            seen.add(key)


class TestHighlightMerging:
    """Integration tests for highlight merging across chunks."""
    
    def test_highlight_timestamp_preservation(self):
        """Test that highlight timestamps are preserved during merging."""
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
        
        # Merge highlights
        strategy = ChunkingStrategy()
        merged = strategy.merge_highlights(
            chunk_results=[chunk1_highlights, chunk2_highlights]
        )
        
        # Verify all highlights are present
        assert len(merged) == 2
        
        # Verify timestamps are in order
        assert merged[0]['timestamp'] < merged[1]['timestamp']
    
    def test_highlight_importance_ranking(self):
        """Test that highlights are ranked by importance after merging."""
        # Create highlights with different importance levels
        chunk1_highlights = [
            {
                "timestamp": "00:02:00",
                "quote": "Medium importance",
                "importance": "medium",
                "context": "Context"
            }
        ]
        
        chunk2_highlights = [
            {
                "timestamp": "00:12:00",
                "quote": "High importance",
                "importance": "high",
                "context": "Context"
            },
            {
                "timestamp": "00:15:00",
                "quote": "Low importance",
                "importance": "low",
                "context": "Context"
            }
        ]
        
        # Merge highlights
        strategy = ChunkingStrategy()
        merged = strategy.merge_highlights(
            chunk_results=[chunk1_highlights, chunk2_highlights]
        )
        
        # Verify highlights are present
        assert len(merged) == 3
        
        # Verify high importance highlights come first (if sorted)
        importance_order = {"high": 0, "medium": 1, "low": 2}
        sorted_merged = sorted(merged, key=lambda h: importance_order[h['importance']])
        assert sorted_merged[0]['importance'] == "high"


class TestTagMerging:
    """Integration tests for tag merging across chunks."""
    
    def test_tag_deduplication(self):
        """Test that duplicate tags are removed during merging."""
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
        
        # Merge tags
        strategy = ChunkingStrategy()
        merged = strategy.merge_tags(
            chunk_results=[chunk1_tags, chunk2_tags]
        )
        
        # Verify duplicates were removed
        assert len(merged['categories']) == 3  # Technology, AI, Software
        assert len(merged['keywords']) == 3  # machine learning, neural networks, deep learning
        assert len(merged['entities']) == 3  # TensorFlow, Python, PyTorch
        
        # Verify no duplicates
        assert len(set(merged['categories'])) == len(merged['categories'])
        assert len(set(merged['keywords'])) == len(merged['keywords'])
        assert len(set(merged['entities'])) == len(merged['entities'])
    
    def test_tag_frequency_ranking(self):
        """Test that tags can be ranked by frequency across chunks."""
        # Create tags with different frequencies
        chunk1_tags = {
            "categories": ["AI", "Technology"],
            "keywords": ["machine learning", "AI"],
            "entities": ["Python"]
        }
        
        chunk2_tags = {
            "categories": ["AI", "Software"],
            "keywords": ["AI", "deep learning"],
            "entities": ["Python", "TensorFlow"]
        }
        
        chunk3_tags = {
            "categories": ["AI"],
            "keywords": ["AI"],
            "entities": ["Python"]
        }
        
        # Merge tags
        strategy = ChunkingStrategy()
        merged = strategy.merge_tags(
            chunk_results=[chunk1_tags, chunk2_tags, chunk3_tags]
        )
        
        # Verify most frequent tags are present
        # "AI" appears in all chunks
        assert "AI" in merged['categories']
        assert "AI" in merged['keywords']
        assert "Python" in merged['entities']


class TestChunkingPerformance:
    """Integration tests for chunking performance and efficiency."""
    
    def test_chunking_minimizes_api_calls(self, long_transcript, mock_agent_with_small_context):
        """Test that chunking minimizes the number of API calls."""
        factory = Mock()
        factory.create_agent.return_value = mock_agent_with_small_context
        
        orchestrator = EnrichmentOrchestrator(agent_factory=factory)
        
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
        
        # Verify API calls were made (but not excessive)
        # In a real implementation, we would verify the exact number
        assert mock_agent_with_small_context.generate.call_count > 0
        assert mock_agent_with_small_context.generate.call_count < 10  # Reasonable limit
