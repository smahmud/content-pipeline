"""
YouTube description generator.

Generates YouTube video descriptions with title, summary, timestamps,
tags, and links optimized for YouTube's format.

Supports platforms: YouTube
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("youtube")
class YouTubeGenerator(BaseGenerator):
    """Generator for YouTube description output format.
    
    Creates YouTube-optimized descriptions with:
    - Video title
    - Summary/description
    - Timestamps (from chapters)
    - Tags/keywords
    - Links section
    
    Required enrichments: summary
    Optional enrichments: chapters, tags, highlights
    """
    
    @property
    def output_type(self) -> str:
        return "youtube"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["youtube"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for YouTube template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        chapters = enriched_content.get("chapters", [])
        highlights = enriched_content.get("highlights", [])
        
        # Extract tags (YouTube allows up to 15)
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        tags = tags[:15] if tags else []
        
        # Build timestamps from chapters
        timestamps = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                timestamp = {
                    "time": chapter.get("start_time", 0),
                    "title": chapter.get("title", chapter.get("summary", "")),
                }
                timestamps.append(timestamp)
        
        # Get title from metadata or generate from summary
        title = self._get_field(enriched_content, "metadata.title", "")
        if not title and summary:
            title = summary.get("short", "")[:100]
        
        # Build key points for description
        key_points = []
        if highlights:
            key_points = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        return {
            "title": title,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "timestamps": timestamps,
            "chapters": chapters,
            "key_points": key_points,
            "highlights": highlights,
            "tags": tags,
            "platform": request.platform or "youtube",
            "tone": request.tone or "friendly",
            "length": request.length or "medium",
            "metadata": enriched_content.get("metadata", {}),
        }
