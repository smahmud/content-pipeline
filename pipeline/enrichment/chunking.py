"""
Chunking Strategy

Handles splitting of long transcripts that exceed LLM context windows.
Implements intelligent splitting at natural boundaries and result merging.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import json

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest


@dataclass
class TextChunk:
    """A chunk of text with metadata.
    
    Attributes:
        text: The chunk text
        start_index: Starting character index in original text
        end_index: Ending character index in original text
        chunk_number: Sequential chunk number (0-indexed)
        total_chunks: Total number of chunks
    """
    text: str
    start_index: int
    end_index: int
    chunk_number: int
    total_chunks: int


class ChunkingStrategy:
    """Handles splitting and merging of long transcripts.
    
    This class implements intelligent chunking that:
    - Detects when transcripts exceed context windows
    - Splits at natural boundaries (paragraphs, sentences)
    - Ensures chunks fit within token limits
    - Merges results coherently
    """
    
    # Safety margin for token counting (90% of context window)
    SAFETY_MARGIN = 0.9
    
    # Minimum chunk size to avoid too many small chunks
    MIN_CHUNK_SIZE = 1000  # characters
    
    def __init__(self, agent: BaseLLMAgent):
        """Initialize chunking strategy.
        
        Args:
            agent: LLM agent to use for token counting and context window detection
        """
        self.agent = agent
    
    def needs_chunking(
        self,
        text: str,
        model: str,
        prompt_overhead: int = 500
    ) -> bool:
        """Check if text needs to be chunked.
        
        Args:
            text: Text to check
            model: Model identifier
            prompt_overhead: Estimated tokens for prompt formatting
            
        Returns:
            True if chunking is needed
        """
        # Get context window for model
        context_window = self.agent.get_context_window(model)
        
        # Estimate tokens in text (rough approximation)
        estimated_tokens = len(text.split()) * 1.3 + prompt_overhead
        
        # Check if it exceeds safe limit
        safe_limit = context_window * self.SAFETY_MARGIN
        return estimated_tokens > safe_limit
    
    def chunk_text(
        self,
        text: str,
        model: str,
        prompt_overhead: int = 500
    ) -> List[TextChunk]:
        """Split text into chunks that fit within context window.
        
        Args:
            text: Text to chunk
            model: Model identifier
            prompt_overhead: Estimated tokens for prompt formatting
            
        Returns:
            List of text chunks
        """
        # Get context window and calculate target chunk size
        context_window = self.agent.get_context_window(model)
        safe_limit = int(context_window * self.SAFETY_MARGIN) - prompt_overhead
        
        # Convert token limit to approximate character limit
        # Assuming ~4 characters per token
        target_chars = safe_limit * 4
        
        # Split at paragraph boundaries first
        paragraphs = self._split_paragraphs(text)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        start_index = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If single paragraph exceeds limit, split at sentences
            if para_size > target_chars:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=start_index + len(chunk_text),
                        chunk_number=len(chunks),
                        total_chunks=0  # Will update later
                    ))
                    start_index += len(chunk_text)
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph at sentences
                sentences = self._split_sentences(para)
                for sentence in sentences:
                    if current_size + len(sentence) > target_chars and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append(TextChunk(
                            text=chunk_text,
                            start_index=start_index,
                            end_index=start_index + len(chunk_text),
                            chunk_number=len(chunks),
                            total_chunks=0
                        ))
                        start_index += len(chunk_text)
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += len(sentence)
            
            # Normal paragraph handling
            elif current_size + para_size > target_chars and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=start_index + len(chunk_text),
                    chunk_number=len(chunks),
                    total_chunks=0
                ))
                start_index += len(chunk_text)
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_index=start_index,
                end_index=start_index + len(chunk_text),
                chunk_number=len(chunks),
                total_chunks=0
            ))
        
        # Update total_chunks for all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def merge_summaries(
        self,
        chunk_summaries: List[Dict[str, str]],
        agent: BaseLLMAgent,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """Merge summaries from multiple chunks.
        
        Args:
            chunk_summaries: List of summary dicts from each chunk
            agent: LLM agent to use for merging
            model: Optional specific model
            
        Returns:
            Merged summary dict with short, medium, long variants
        """
        # If only one chunk, return as-is
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        
        # Combine all summaries
        combined_short = " ".join(s.get("short", "") for s in chunk_summaries)
        combined_medium = " ".join(s.get("medium", "") for s in chunk_summaries)
        combined_long = "\n\n".join(s.get("long", "") for s in chunk_summaries)
        
        # Use LLM to create coherent merged summaries
        merge_prompt = f"""You are merging summaries from {len(chunk_summaries)} chunks of a transcript.
Create coherent, unified summaries at three lengths.

Chunk summaries:
{json.dumps(chunk_summaries, indent=2)}

Provide your response in JSON format:
{{
  "short": "1-2 sentence unified summary",
  "medium": "Paragraph-length unified summary",
  "long": "Detailed multi-paragraph unified summary"
}}"""
        
        request = LLMRequest(
            prompt=merge_prompt,
            max_tokens=1000,
            temperature=0.3,
            model=model
        )
        
        try:
            response = agent.generate(request)
            merged = json.loads(response.content)
            return merged
        except Exception:
            # Fallback: return combined summaries
            return {
                "short": combined_short[:500],
                "medium": combined_medium[:2000],
                "long": combined_long[:5000]
            }
    
    def merge_tags(
        self,
        chunk_tags: List[Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        """Merge tags from multiple chunks.
        
        Args:
            chunk_tags: List of tag dicts from each chunk
            
        Returns:
            Merged tag dict with deduplicated lists
        """
        # Combine and deduplicate
        all_categories = set()
        all_keywords = set()
        all_entities = set()
        
        for tags in chunk_tags:
            all_categories.update(tags.get("categories", []))
            all_keywords.update(tags.get("keywords", []))
            all_entities.update(tags.get("entities", []))
        
        return {
            "categories": sorted(list(all_categories)),
            "keywords": sorted(list(all_keywords)),
            "entities": sorted(list(all_entities))
        }
    
    def merge_chapters(
        self,
        chunk_chapters: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Merge chapters from multiple chunks.
        
        Args:
            chunk_chapters: List of chapter lists from each chunk
            
        Returns:
            Merged and deduplicated chapter list
        """
        # Flatten all chapters
        all_chapters = []
        for chapters in chunk_chapters:
            all_chapters.extend(chapters)
        
        # Sort by start time
        all_chapters.sort(key=lambda c: c.get("start_time", "00:00:00"))
        
        # Remove duplicates based on similar start times
        merged = []
        for chapter in all_chapters:
            # Check if similar chapter already exists
            is_duplicate = False
            for existing in merged:
                if self._similar_timestamps(
                    chapter.get("start_time", ""),
                    existing.get("start_time", "")
                ):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(chapter)
        
        return merged
    
    def merge_highlights(
        self,
        chunk_highlights: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Merge highlights from multiple chunks.
        
        Args:
            chunk_highlights: List of highlight lists from each chunk
            
        Returns:
            Merged and deduplicated highlight list
        """
        # Flatten all highlights
        all_highlights = []
        for highlights in chunk_highlights:
            all_highlights.extend(highlights)
        
        # Sort by timestamp
        all_highlights.sort(key=lambda h: h.get("timestamp", "00:00:00"))
        
        # Remove duplicates based on similar timestamps
        merged = []
        for highlight in all_highlights:
            # Check if similar highlight already exists
            is_duplicate = False
            for existing in merged:
                if self._similar_timestamps(
                    highlight.get("timestamp", ""),
                    existing.get("timestamp", "")
                ):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(highlight)
        
        return merged
    
    def _similar_timestamps(self, ts1: str, ts2: str, threshold: int = 5) -> bool:
        """Check if two timestamps are within threshold seconds.
        
        Args:
            ts1: First timestamp (HH:MM:SS)
            ts2: Second timestamp (HH:MM:SS)
            threshold: Threshold in seconds
            
        Returns:
            True if timestamps are similar
        """
        try:
            # Parse timestamps
            def parse_ts(ts: str) -> int:
                parts = ts.split(":")
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + int(s)
                return 0
            
            seconds1 = parse_ts(ts1)
            seconds2 = parse_ts(ts2)
            
            return abs(seconds1 - seconds2) <= threshold
        except Exception:
            return False
