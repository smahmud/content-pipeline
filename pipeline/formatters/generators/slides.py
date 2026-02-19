"""
Presentation slides generator.

Generates slide deck outlines with title slide, content slides,
and conclusion slide in Markdown format.

Supports platforms: General presentation formats
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("slides")
class SlidesGenerator(BaseGenerator):
    """Generator for presentation slides output format.
    
    Creates slide deck outline with:
    - Title slide
    - Agenda/overview slide
    - Content slides (from chapters/highlights)
    - Key takeaways slide
    - Conclusion/CTA slide
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "slides"
    
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
        """Build context for slides template.
        
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
        title = self._get_field(enriched_content, "metadata.title", "Presentation")
        
        # Build content slides from chapters or highlights
        content_slides = []
        
        if chapters:
            for chapter in chapters[:8]:  # Limit to 8 content slides
                if isinstance(chapter, dict):
                    content_slides.append({
                        "title": chapter.get("title", ""),
                        "content": chapter.get("summary", ""),
                        "bullet_points": [],
                    })
        elif highlights:
            # Group highlights into slides (2-3 per slide)
            for i in range(0, len(highlights[:9]), 3):
                slide_highlights = highlights[i:i+3]
                bullet_points = [
                    h.get("text", h) if isinstance(h, dict) else h 
                    for h in slide_highlights
                ]
                content_slides.append({
                    "title": f"Key Points {i//3 + 1}",
                    "content": "",
                    "bullet_points": bullet_points,
                })
        
        # Build key takeaways
        key_takeaways = []
        if highlights:
            key_takeaways = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        # Extract tags for topics
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        return {
            "title": title,
            "subtitle": summary.get("short", "")[:100],
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "content_slides": content_slides,
            "key_takeaways": key_takeaways,
            "chapters": chapters,
            "highlights": highlights,
            "tags": tags[:5],
            "slide_count": len(content_slides) + 3,  # +3 for title, agenda, conclusion
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
