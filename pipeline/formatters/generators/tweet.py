"""
Tweet/Twitter thread generator.

Generates tweet content with hook, main points, and conclusion
optimized for Twitter's character limits.

Supports platforms: Twitter
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("tweet")
class TweetGenerator(BaseGenerator):
    """Generator for tweet/Twitter thread output format.
    
    Creates Twitter-optimized content with:
    - Hook/attention grabber
    - Main points (thread format)
    - Conclusion with CTA
    - Hashtags
    
    Required enrichments: summary
    Optional enrichments: tags, highlights
    """
    
    @property
    def output_type(self) -> str:
        return "tweet"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["twitter"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for tweet template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        highlights = enriched_content.get("highlights", [])
        
        # Extract tags for hashtags (limit to 5 for Twitter)
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", [])[:3] + tags.get("secondary", [])[:2]
        hashtags = tags[:5] if tags else []
        
        # Build thread points from highlights or summary
        thread_points = []
        if highlights:
            thread_points = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        # Get short summary for hook
        hook = summary.get("short", "")
        if not hook and summary.get("medium"):
            # Take first sentence of medium summary
            hook = summary.get("medium", "").split(".")[0] + "."
        
        return {
            "hook": hook,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "thread_points": thread_points,
            "highlights": highlights,
            "hashtags": hashtags,
            "tags": tags,
            "platform": request.platform or "twitter",
            "tone": request.tone or "casual",
            "length": request.length or "short",
            "metadata": enriched_content.get("metadata", {}),
        }
