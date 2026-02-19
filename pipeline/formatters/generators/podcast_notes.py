"""
Podcast show notes generator.

Generates podcast show notes with episode summary, key topics,
timestamps, and resources mentioned.

Supports platforms: Podcast hosting platforms
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("podcast-notes")
class PodcastNotesGenerator(BaseGenerator):
    """Generator for podcast show notes output format.
    
    Creates podcast show notes with:
    - Episode title and summary
    - Key topics discussed
    - Timestamps/chapters
    - Resources and links mentioned
    - Guest information (if available)
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "podcast-notes"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["medium", "wordpress", "ghost", "substack"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for podcast notes template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        chapters = enriched_content.get("chapters", [])
        highlights = enriched_content.get("highlights", [])
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        
        # Build key topics from highlights or tags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        key_topics = []
        if highlights:
            key_topics = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:7]
            ]
        elif tags:
            key_topics = tags[:7]
        
        # Process chapters for timestamps
        timestamps = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                timestamps.append({
                    "time": chapter.get("start_time", 0),
                    "title": chapter.get("title", ""),
                    "summary": chapter.get("summary", ""),
                })
        
        return {
            "title": title,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "key_topics": key_topics,
            "timestamps": timestamps,
            "chapters": chapters,
            "highlights": highlights,
            "tags": tags,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
