"""
Chapters/timestamps generator.

Generates chapter markers with timestamps, titles, and descriptions
for video or audio content.

Supports platforms: YouTube, podcast platforms
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("chapters")
class ChaptersGenerator(BaseGenerator):
    """Generator for chapters/timestamps output format.
    
    Creates chapter markers with:
    - Timestamps (HH:MM:SS or MM:SS)
    - Chapter titles
    - Optional descriptions
    
    Required enrichments: chapters
    Optional enrichments: summary
    """
    
    @property
    def output_type(self) -> str:
        return "chapters"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["youtube"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["chapters"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for chapters template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        chapters = enriched_content.get("chapters", [])
        summary = enriched_content.get("summary", {})
        
        # Process chapters to ensure consistent format
        processed_chapters = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                processed_chapters.append({
                    "start_time": chapter.get("start_time", 0),
                    "end_time": chapter.get("end_time"),
                    "title": chapter.get("title", ""),
                    "summary": chapter.get("summary", ""),
                })
            elif isinstance(chapter, str):
                # Handle string format "00:00 Title"
                parts = chapter.split(" ", 1)
                if len(parts) == 2:
                    processed_chapters.append({
                        "start_time": parts[0],
                        "title": parts[1],
                        "summary": "",
                    })
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        
        return {
            "title": title,
            "chapters": processed_chapters,
            "chapter_count": len(processed_chapters),
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "platform": request.platform or "youtube",
            "metadata": enriched_content.get("metadata", {}),
        }
