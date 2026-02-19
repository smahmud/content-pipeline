"""
Obsidian note generator.

Generates Obsidian-formatted notes with YAML frontmatter,
wiki-style links, tags, and structured sections.

Supports platforms: Obsidian
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("obsidian")
class ObsidianGenerator(BaseGenerator):
    """Generator for Obsidian note output format.
    
    Creates Obsidian-optimized notes with:
    - YAML frontmatter (tags, aliases, date)
    - Wiki-style internal links
    - Structured sections with headers
    - Tag formatting (#tag)
    - Callouts for highlights
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights, tags
    """
    
    @property
    def output_type(self) -> str:
        return "obsidian"
    
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
        """Build context for Obsidian template.
        
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
        
        # Extract tags and format for Obsidian
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            tags = tags.get("primary", []) + tags.get("secondary", [])
        
        # Format tags for Obsidian (lowercase, no spaces)
        obsidian_tags = []
        for tag in tags[:10]:
            formatted = tag.lower().replace(" ", "-").replace("_", "-")
            obsidian_tags.append(formatted)
        
        # Build sections from chapters
        sections = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                sections.append({
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
        
        # Build aliases from title variations
        aliases = []
        if title:
            # Add shortened version as alias
            short_title = title.split(":")[0].strip()
            if short_title != title:
                aliases.append(short_title)
        
        return {
            "title": title,
            "date": date,
            "source_url": source_url,
            "aliases": aliases,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "long_summary": summary.get("long", ""),
            "sections": sections,
            "callout_highlights": callout_highlights,
            "tags": tags,
            "obsidian_tags": obsidian_tags,
            "chapters": chapters,
            "highlights": highlights,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
