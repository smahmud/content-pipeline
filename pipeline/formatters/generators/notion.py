"""
Notion page generator.

Generates Notion-formatted content with database properties,
toggles, callouts, and structured sections.

Supports platforms: Notion
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("notion")
class NotionGenerator(BaseGenerator):
    """Generator for Notion page output format.
    
    Creates Notion-optimized content with:
    - Page title and icon
    - Database properties (tags, status, date)
    - Summary callout
    - Toggle sections for chapters
    - Highlights as callouts
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "notion"
    
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
        """Build context for Notion template.
        
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
        
        # Extract tags for properties
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        # Build toggle sections from chapters
        toggle_sections = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                toggle_sections.append({
                    "title": chapter.get("title", ""),
                    "content": chapter.get("summary", ""),
                })
        
        # Build callout highlights
        callout_highlights = []
        if highlights:
            callout_highlights = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        # Get date from metadata
        date = self._get_field(enriched_content, "metadata.date", "")
        if not date:
            date = self._get_field(enriched_content, "metadata.timestamp", "")
        
        # Get source URL
        source_url = self._get_field(enriched_content, "metadata.url", "")
        
        return {
            "title": title,
            "icon": "📝",  # Default icon
            "date": date,
            "source_url": source_url,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "toggle_sections": toggle_sections,
            "callout_highlights": callout_highlights,
            "tags": tags,
            "chapters": chapters,
            "highlights": highlights,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
