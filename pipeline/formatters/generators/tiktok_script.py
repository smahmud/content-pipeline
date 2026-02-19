"""
TikTok script generator.

Generates short-form video scripts optimized for TikTok
with hook, body, and CTA within 60-second timing.

Supports platforms: TikTok, Instagram Reels, YouTube Shorts
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("tiktok-script")
class TikTokScriptGenerator(BaseGenerator):
    """Generator for TikTok script output format.
    
    Creates short-form video scripts with:
    - Attention-grabbing hook (3-5 seconds)
    - Main content body (45-50 seconds)
    - Call-to-action (5-10 seconds)
    - Total duration: ~60 seconds
    
    Required enrichments: summary
    Optional enrichments: highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "tiktok-script"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["twitter", "linkedin"]  # For sharing script
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for TikTok script template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        highlights = enriched_content.get("highlights", [])
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        
        # Build hook from short summary (attention grabber)
        hook = summary.get("short", "")
        if hook:
            # Make it punchy - take first sentence or question
            first_sentence = hook.split(".")[0]
            if len(first_sentence) > 50:
                first_sentence = first_sentence[:50] + "..."
            hook = first_sentence
        
        # Build main points from highlights (3 max for 60 seconds)
        main_points = []
        if highlights:
            main_points = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:3]
            ]
        
        # Extract tags for hashtags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", [])[:3] + tags.get("secondary", [])[:2]
        hashtags = tags[:5] if tags else []
        
        # Build script sections with timing
        script_sections = [
            {
                "section": "hook",
                "timing": "0-5 seconds",
                "content": hook,
                "visual": "Face close-up, energetic",
            },
            {
                "section": "body",
                "timing": "5-50 seconds",
                "content": summary.get("medium", summary.get("short", "")),
                "points": main_points,
                "visual": "Dynamic cuts, text overlays",
            },
            {
                "section": "cta",
                "timing": "50-60 seconds",
                "content": "Follow for more! Drop a comment below!",
                "visual": "Point to follow button",
            },
        ]
        
        return {
            "title": title,
            "hook": hook,
            "main_points": main_points,
            "script_sections": script_sections,
            "total_duration": "60 seconds",
            "hashtags": hashtags,
            "tags": tags,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "highlights": highlights,
            "platform": request.platform or "tiktok",
            "tone": request.tone or "casual",
            "metadata": enriched_content.get("metadata", {}),
        }
