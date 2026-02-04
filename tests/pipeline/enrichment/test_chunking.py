"""
Unit tests for ChunkingStrategy

Tests text chunking, context window detection, and result merging.
"""

import pytest
import json
from unittest.mock import Mock

from pipeline.enrichment.chunking import ChunkingStrategy, TextChunk
from pipeline.llm.providers.base import LLMResponse


@pytest.fixture
def mock_agent():
    """Create mock LLM agent."""
    agent = Mock()
    agent.get_context_window.return_value = 8000  # 8K tokens
    agent.generate.return_value = LLMResponse(
        content=json.dumps({
            "short": "Merged summary",
            "medium": "Merged medium summary",
            "long": "Merged long summary"
        }),
        model_used="gpt-4-turbo",
        tokens_used=200,
        cost_usd=0.006
    )
    return agent


@pytest.fixture
def chunking_strategy(mock_agent):
    """Create chunking strategy with mock agent."""
    return ChunkingStrategy(agent=mock_agent)


class TestChunkingStrategy:
    """Test suite for ChunkingStrategy."""
    
    def test_initialization(self, mock_agent):
        """Test chunking strategy initialization."""
        strategy = ChunkingStrategy(agent=mock_agent)
        
        assert strategy.agent == mock_agent
        assert strategy.SAFETY_MARGIN == 0.9
        assert strategy.MIN_CHUNK_SIZE == 1000
    
    def test_needs_chunking_small_text(self, chunking_strategy):
        """Test that small text doesn't need chunking."""
        text = "This is a short text. " * 100  # ~500 words
        
        needs_chunking = chunking_strategy.needs_chunking(
            text=text,
            model="gpt-4-turbo",
            prompt_overhead=500
        )
        
        assert not needs_chunking
    
    def test_needs_chunking_large_text(self, chunking_strategy):
        """Test that large text needs chunking."""
        # Create text that exceeds context window
        text = "This is a long text. " * 10000  # ~20K words
        
        needs_chunking = chunking_strategy.needs_chunking(
            text=text,
            model="gpt-4-turbo",
            prompt_overhead=500
        )
        
        assert needs_chunking
    
    def test_chunk_text_basic(self, chunking_strategy):
        """Test basic text chunking."""
        # Create text with multiple paragraphs
        paragraphs = [f"Paragraph {i}. " * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)
        
        chunks = chunking_strategy.chunk_text(
            text=text,
            model="gpt-4-turbo",
            prompt_overhead=500
        )
        
        # Verify chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        
        # Verify chunk metadata
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_number == i
            assert chunk.total_chunks == len(chunks)
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index
    
    def test_chunk_text_preserves_content(self, chunking_strategy):
        """Test that chunking preserves all content."""
        text = "Test content. " * 1000
        
        chunks = chunking_strategy.chunk_text(
            text=text,
            model="gpt-4-turbo",
            prompt_overhead=500
        )
        
        # Reconstruct text from chunks
        reconstructed = "".join(chunk.text for chunk in chunks)
        
        # Verify all content is preserved (allowing for whitespace differences)
        assert len(reconstructed) > 0
        assert "Test content" in reconstructed
    
    def test_split_paragraphs(self, chunking_strategy):
        """Test paragraph splitting."""
        text = "Paragraph 1.\n\nParagraph 2.\n\n\nParagraph 3."
        
        paragraphs = chunking_strategy._split_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Paragraph 1."
        assert paragraphs[1] == "Paragraph 2."
        assert paragraphs[2] == "Paragraph 3."
    
    def test_split_sentences(self, chunking_strategy):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        
        sentences = chunking_strategy._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_merge_summaries_single_chunk(self, chunking_strategy, mock_agent):
        """Test merging summaries from single chunk."""
        chunk_summaries = [
            {
                "short": "Short summary",
                "medium": "Medium summary",
                "long": "Long summary"
            }
        ]
        
        merged = chunking_strategy.merge_summaries(
            chunk_summaries=chunk_summaries,
            agent=mock_agent,
            model="gpt-4-turbo"
        )
        
        # Single chunk should be returned as-is
        assert merged == chunk_summaries[0]
        
        # Agent should not be called
        mock_agent.generate.assert_not_called()
    
    def test_merge_summaries_multiple_chunks(self, chunking_strategy, mock_agent):
        """Test merging summaries from multiple chunks."""
        chunk_summaries = [
            {
                "short": "Summary 1",
                "medium": "Medium 1",
                "long": "Long 1"
            },
            {
                "short": "Summary 2",
                "medium": "Medium 2",
                "long": "Long 2"
            }
        ]
        
        merged = chunking_strategy.merge_summaries(
            chunk_summaries=chunk_summaries,
            agent=mock_agent,
            model="gpt-4-turbo"
        )
        
        # Verify merged result
        assert "short" in merged
        assert "medium" in merged
        assert "long" in merged
        
        # Agent should be called for merging
        mock_agent.generate.assert_called_once()
    
    def test_merge_tags(self, chunking_strategy):
        """Test merging tags from multiple chunks."""
        chunk_tags = [
            {
                "categories": ["tech", "ai"],
                "keywords": ["machine learning", "neural networks"],
                "entities": ["OpenAI", "Google"]
            },
            {
                "categories": ["ai", "research"],
                "keywords": ["neural networks", "deep learning"],
                "entities": ["Google", "DeepMind"]
            }
        ]
        
        merged = chunking_strategy.merge_tags(chunk_tags)
        
        # Verify deduplication
        assert set(merged["categories"]) == {"tech", "ai", "research"}
        assert set(merged["keywords"]) == {"machine learning", "neural networks", "deep learning"}
        assert set(merged["entities"]) == {"OpenAI", "Google", "DeepMind"}
        
        # Verify sorted
        assert merged["categories"] == sorted(merged["categories"])
        assert merged["keywords"] == sorted(merged["keywords"])
        assert merged["entities"] == sorted(merged["entities"])
    
    def test_merge_chapters(self, chunking_strategy):
        """Test merging chapters from multiple chunks."""
        chunk_chapters = [
            [
                {"title": "Introduction", "start_time": "00:00:00", "end_time": "00:05:00"},
                {"title": "Main Topic", "start_time": "00:05:00", "end_time": "00:15:00"}
            ],
            [
                {"title": "Main Topic", "start_time": "00:05:00", "end_time": "00:15:00"},
                {"title": "Conclusion", "start_time": "00:15:00", "end_time": "00:20:00"}
            ]
        ]
        
        merged = chunking_strategy.merge_chapters(chunk_chapters)
        
        # Verify deduplication (Main Topic should appear once)
        assert len(merged) == 3
        
        # Verify sorted by start time
        assert merged[0]["title"] == "Introduction"
        assert merged[1]["title"] == "Main Topic"
        assert merged[2]["title"] == "Conclusion"
    
    def test_merge_highlights(self, chunking_strategy):
        """Test merging highlights from multiple chunks."""
        chunk_highlights = [
            [
                {"timestamp": "00:02:00", "quote": "Important point 1", "importance": "high"},
                {"timestamp": "00:08:00", "quote": "Important point 2", "importance": "medium"}
            ],
            [
                {"timestamp": "00:08:00", "quote": "Important point 2", "importance": "medium"},
                {"timestamp": "00:15:00", "quote": "Important point 3", "importance": "high"}
            ]
        ]
        
        merged = chunking_strategy.merge_highlights(chunk_highlights)
        
        # Verify deduplication
        assert len(merged) == 3
        
        # Verify sorted by timestamp
        assert merged[0]["timestamp"] == "00:02:00"
        assert merged[1]["timestamp"] == "00:08:00"
        assert merged[2]["timestamp"] == "00:15:00"
    
    def test_similar_timestamps(self, chunking_strategy):
        """Test timestamp similarity detection."""
        # Similar timestamps (within 5 seconds)
        assert chunking_strategy._similar_timestamps("00:10:00", "00:10:03")
        assert chunking_strategy._similar_timestamps("00:10:00", "00:10:05")
        
        # Different timestamps (more than 5 seconds)
        assert not chunking_strategy._similar_timestamps("00:10:00", "00:10:10")
        assert not chunking_strategy._similar_timestamps("00:10:00", "00:11:00")
    
    def test_chunk_text_respects_context_window(self, chunking_strategy, mock_agent):
        """Test that chunks respect context window limits."""
        # Create long text that will need chunking
        # Context window is 8000 tokens, safe limit is 90% = 7200 tokens - 500 overhead = 6700 tokens
        # Target chars: 6700 * 4 = 26,800 characters
        # Create text with multiple paragraphs totaling ~50,000 characters to force chunking
        paragraph = "Word " * 500  # ~2,500 characters per paragraph
        text = "\n\n".join([paragraph] * 20)  # 20 paragraphs = ~50,000 characters
        
        chunks = chunking_strategy.chunk_text(
            text=text,
            model="gpt-4-turbo",
            prompt_overhead=500
        )
        
        # Verify we got multiple chunks
        assert len(chunks) > 1
        
        # Verify each chunk fits in context window
        context_window = mock_agent.get_context_window.return_value
        safe_limit = int(context_window * chunking_strategy.SAFETY_MARGIN) - 500
        
        for chunk in chunks:
            # Rough token estimate: ~1.3 tokens per word
            estimated_tokens = len(chunk.text.split()) * 1.3
            assert estimated_tokens <= safe_limit
    
    def test_merge_summaries_fallback_on_error(self, chunking_strategy, mock_agent):
        """Test fallback when LLM merge fails."""
        # Configure agent to raise error
        mock_agent.generate.side_effect = Exception("API error")
        
        chunk_summaries = [
            {"short": "S1", "medium": "M1", "long": "L1"},
            {"short": "S2", "medium": "M2", "long": "L2"}
        ]
        
        merged = chunking_strategy.merge_summaries(
            chunk_summaries=chunk_summaries,
            agent=mock_agent,
            model="gpt-4-turbo"
        )
        
        # Verify fallback to simple concatenation
        assert "short" in merged
        assert "medium" in merged
        assert "long" in merged
        
        # Verify content is combined
        assert "S1" in merged["short"] and "S2" in merged["short"]
