"""
Quote cards generator.

Generates shareable quote cards from highlights with
attribution and formatting for social media.

Supports platforms: Twitter, LinkedIn, Instagram
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("quote-cards")
class QuoteCardsGenerator(BaseGenerator):
    """Generator for quote cards output format.
    
    Creates shareable quote cards with:
    - Quote text (from highlights)
    - Attribution/source
    - Hashtags
    - Platform-specific formatting
    
    Required enrichments: highlights
    Optional enrichments: tags, summary
    """
    
    @property
    def output_type(self) -> str:
        return "quote-cards"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["twitter", "linkedin"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["highlights"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for quote cards template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        highlights = enriched_content.get("highlights", [])
        summary = enriched_content.get("summary", {})
        
        # Get title/source from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        author = self._get_field(enriched_content, "metadata.author", "")
        source_url = self._get_field(enriched_content, "metadata.url", "")
        
        # Extract tags for hashtags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", [])[:3] + tags.get("secondary", [])[:2]
        hashtags = tags[:5] if tags else []
        
        # Build quote cards from highlights
        quote_cards = []
        for i, highlight in enumerate(highlights[:10]):  # Limit to 10 quotes
            text = highlight.get("text", highlight) if isinstance(highlight, dict) else highlight
            
            # Get timestamp if available
            timestamp = None
            if isinstance(highlight, dict):
                timestamp = highlight.get("timestamp") or highlight.get("start_time")
            
            quote_cards.append({
                "index": i + 1,
                "text": text,
                "timestamp": timestamp,
                "attribution": author or title,
            })
        
        return {
            "title": title,
            "author": author,
            "source_url": source_url,
            "quote_cards": quote_cards,
            "quote_count": len(quote_cards),
            "hashtags": hashtags,
            "tags": tags,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "highlights": highlights,
            "platform": request.platform or "twitter",
            "metadata": enriched_content.get("metadata", {}),
        }
