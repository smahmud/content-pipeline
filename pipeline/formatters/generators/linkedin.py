"""
LinkedIn post generator.

Generates LinkedIn-optimized posts with professional hook,
key insights, and engagement prompts.

Supports platforms: LinkedIn
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("linkedin")
class LinkedInGenerator(BaseGenerator):
    """Generator for LinkedIn post output format.
    
    Creates LinkedIn-optimized posts with:
    - Professional hook
    - Key insights/takeaways
    - Engagement prompt/question
    - Relevant hashtags
    
    Required enrichments: summary
    Optional enrichments: highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "linkedin"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["linkedin"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for LinkedIn template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        highlights = enriched_content.get("highlights", [])
        
        # Extract tags for hashtags (LinkedIn allows up to 10)
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", [])[:5] + tags.get("secondary", [])[:5]
        hashtags = tags[:10] if tags else []
        
        # Build key insights from highlights
        insights = []
        if highlights:
            insights = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        # Get hook from short summary
        hook = summary.get("short", "")
        if not hook:
            hook = summary.get("medium", "").split(".")[0] + "." if summary.get("medium") else ""
        
        return {
            "hook": hook,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "insights": insights,
            "highlights": highlights,
            "hashtags": hashtags,
            "tags": tags,
            "platform": request.platform or "linkedin",
            "tone": request.tone or "professional",
            "length": request.length or "medium",
            "metadata": enriched_content.get("metadata", {}),
        }
