"""
Newsletter generator.

Generates email newsletter content with subject line, preview text,
sections, and footer.

Supports platforms: Substack, general email
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("newsletter")
class NewsletterGenerator(BaseGenerator):
    """Generator for newsletter output format.
    
    Creates newsletter content with:
    - Subject line
    - Preview text
    - Introduction
    - Content sections
    - Footer with CTA
    
    Required enrichments: summary
    Optional enrichments: highlights, chapters, tags
    """
    
    @property
    def output_type(self) -> str:
        return "newsletter"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["substack", "medium"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for newsletter template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        highlights = enriched_content.get("highlights", [])
        chapters = enriched_content.get("chapters", [])
        
        # Build subject line from title or short summary
        title = self._get_field(enriched_content, "metadata.title", "")
        subject = title if title else summary.get("short", "")[:60]
        
        # Build preview text (email preview, ~100 chars)
        preview = summary.get("short", "")[:100]
        
        # Build sections from chapters or highlights
        sections = []
        if chapters:
            for chapter in chapters[:5]:
                if isinstance(chapter, dict):
                    sections.append({
                        "title": chapter.get("title", ""),
                        "content": chapter.get("summary", ""),
                    })
        elif highlights:
            for i, highlight in enumerate(highlights[:5]):
                text = highlight.get("text", highlight) if isinstance(highlight, dict) else highlight
                sections.append({
                    "title": f"Key Point {i + 1}",
                    "content": text,
                })
        
        # Extract tags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        return {
            "subject": subject,
            "preview": preview,
            "title": title,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "sections": sections,
            "highlights": highlights,
            "chapters": chapters,
            "tags": tags,
            "platform": request.platform or "substack",
            "tone": request.tone or "friendly",
            "length": request.length or "medium",
            "metadata": enriched_content.get("metadata", {}),
        }
