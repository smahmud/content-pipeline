"""
Blog post generator.

Generates blog post content with title, introduction, body sections,
conclusion, and call-to-action from enriched content.

Supports platforms: Medium, WordPress, Ghost, Substack
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("blog")
class BlogGenerator(BaseGenerator):
    """Generator for blog post output format.
    
    Creates structured blog posts with:
    - Title
    - Introduction/hook
    - Body sections with headers
    - Conclusion
    - Call-to-action
    
    Required enrichments: summary, tags
    Optional enrichments: highlights, chapters
    """
    
    @property
    def output_type(self) -> str:
        return "blog"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["medium", "wordpress", "ghost", "substack"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary", "tags"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for blog template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        # Extract summary variants
        summary = enriched_content.get("summary", {})
        
        # Extract key points from highlights or chapters
        highlights = enriched_content.get("highlights", [])
        chapters = enriched_content.get("chapters", [])
        
        # Build key points from highlights or chapter titles
        key_points = []
        if highlights:
            key_points = [h.get("text", h) if isinstance(h, dict) else h for h in highlights[:5]]
        elif chapters:
            key_points = [c.get("title", c.get("summary", "")) for c in chapters[:5]]
        
        # Extract tags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        # Get transcript for additional context
        transcript = enriched_content.get("transcript", {})
        transcript_text = ""
        if isinstance(transcript, dict):
            transcript_text = transcript.get("text", "")
        elif isinstance(transcript, str):
            transcript_text = transcript
        
        # Build title from summary or first highlight
        title = self._get_field(enriched_content, "metadata.title", "")
        if not title and summary:
            # Generate title from short summary
            short_summary = summary.get("short", "")
            if short_summary:
                # Take first sentence or first 60 chars
                title = short_summary.split(".")[0][:60]
        
        return {
            "title": title,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "key_points": key_points,
            "highlights": highlights,
            "chapters": chapters,
            "tags": tags,
            "transcript": transcript_text,
            "platform": request.platform,
            "tone": request.tone or "professional",
            "length": request.length or "medium",
            "metadata": enriched_content.get("metadata", {}),
        }
