"""
AI video script generator.

Generates video scripts for AI avatar platforms like HeyGen
with scene markers, dialogue, and timing cues.

Supports platforms: HeyGen, Synthesia, D-ID
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("video-script")
class VideoScriptGenerator(BaseGenerator):
    """Generator for AI video script output format.
    
    Creates video scripts with:
    - Scene markers
    - Dialogue/narration
    - Timing cues
    - Visual suggestions
    - Transitions
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights
    """
    
    @property
    def output_type(self) -> str:
        return "video-script"
    
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
        """Build context for video script template.
        
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
        
        # Build scenes from chapters or create default structure
        scenes = []
        
        if chapters:
            # Opening scene
            scenes.append({
                "number": 1,
                "type": "intro",
                "title": "Introduction",
                "dialogue": summary.get("short", ""),
                "duration": "15-20 seconds",
                "visual": "Speaker facing camera",
            })
            
            # Content scenes from chapters
            for i, chapter in enumerate(chapters[:5], start=2):
                if isinstance(chapter, dict):
                    scenes.append({
                        "number": i,
                        "type": "content",
                        "title": chapter.get("title", f"Section {i-1}"),
                        "dialogue": chapter.get("summary", ""),
                        "duration": "30-45 seconds",
                        "visual": "Speaker with relevant graphics",
                    })
            
            # Closing scene
            scenes.append({
                "number": len(scenes) + 1,
                "type": "outro",
                "title": "Conclusion",
                "dialogue": "Thank you for watching. Don't forget to like and subscribe!",
                "duration": "10-15 seconds",
                "visual": "Speaker with CTA overlay",
            })
        else:
            # Default 3-scene structure
            scenes = [
                {
                    "number": 1,
                    "type": "intro",
                    "title": "Introduction",
                    "dialogue": summary.get("short", ""),
                    "duration": "15-20 seconds",
                    "visual": "Speaker facing camera",
                },
                {
                    "number": 2,
                    "type": "content",
                    "title": "Main Content",
                    "dialogue": summary.get("medium", summary.get("long", "")),
                    "duration": "60-90 seconds",
                    "visual": "Speaker with supporting visuals",
                },
                {
                    "number": 3,
                    "type": "outro",
                    "title": "Conclusion",
                    "dialogue": "Thank you for watching!",
                    "duration": "10-15 seconds",
                    "visual": "Speaker with CTA overlay",
                },
            ]
        
        # Build key points for visual cues
        key_points = []
        if highlights:
            key_points = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:5]
            ]
        
        # Estimate total duration
        total_duration = f"{len(scenes) * 30}-{len(scenes) * 45} seconds"
        
        return {
            "title": title,
            "scenes": scenes,
            "scene_count": len(scenes),
            "total_duration": total_duration,
            "key_points": key_points,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "chapters": chapters,
            "highlights": highlights,
            "platform": request.platform,
            "tone": request.tone or "professional",
            "metadata": enriched_content.get("metadata", {}),
        }
