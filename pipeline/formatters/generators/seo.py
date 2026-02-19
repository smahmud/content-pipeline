"""
SEO metadata generator.

Generates SEO-optimized metadata including meta title, meta description,
keywords, Open Graph tags, and Twitter cards.

Supports platforms: All (platform-agnostic SEO metadata)
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("seo")
class SEOGenerator(BaseGenerator):
    """Generator for SEO metadata output format.
    
    Creates SEO-optimized metadata with:
    - Meta title (≤60 characters)
    - Meta description (≤160 characters)
    - Keywords
    - Open Graph tags
    - Twitter card tags
    
    Required enrichments: summary, tags
    Optional enrichments: highlights
    """
    
    @property
    def output_type(self) -> str:
        return "seo"
    
    @property
    def supported_platforms(self) -> list[str]:
        # SEO is platform-agnostic
        return ["medium", "wordpress", "ghost", "substack"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary", "tags"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for SEO template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        
        # Extract tags for keywords
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            primary_tags = tags.get("primary", [])
            secondary_tags = tags.get("secondary", [])
            all_tags = primary_tags + secondary_tags
        else:
            all_tags = tags if tags else []
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        
        # Build meta title (≤60 chars)
        meta_title = title[:60] if title else ""
        if not meta_title and summary:
            short = summary.get("short", "")
            meta_title = short.split(".")[0][:60] if short else ""
        
        # Build meta description (≤160 chars)
        meta_description = summary.get("short", "")[:160]
        if not meta_description:
            meta_description = summary.get("medium", "")[:160]
        
        # Build keywords (comma-separated)
        keywords = all_tags[:10]  # Limit to 10 keywords
        
        # Get image URL if available
        image_url = self._get_field(enriched_content, "metadata.image_url", "")
        
        # Get author if available
        author = self._get_field(enriched_content, "metadata.author", "")
        
        # Get URL/canonical if available
        canonical_url = self._get_field(enriched_content, "metadata.url", "")
        
        return {
            "title": title,
            "meta_title": meta_title,
            "meta_description": meta_description,
            "keywords": keywords,
            "keywords_string": ", ".join(keywords),
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "tags": all_tags,
            "primary_tags": tags.get("primary", []) if isinstance(tags, dict) else all_tags[:5],
            "secondary_tags": tags.get("secondary", []) if isinstance(tags, dict) else all_tags[5:],
            "image_url": image_url,
            "author": author,
            "canonical_url": canonical_url,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
