"""
AI video script generator for AI video generation tools.

Generates scene-by-scene video scripts for AI video generators
like Sora, Runway, and Pika. Unlike video-script (for avatar platforms),
this produces visual prompts and voiceover text for fully AI-generated video.

Supports platforms: YouTube, TikTok, Vimeo
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory
from pipeline.formatters.schemas.video_script import (
    PLATFORM_CONFIGS,
    WORDS_PER_MINUTE,
)


@GeneratorFactory.register("ai-video-script")
class AIVideoScriptGenerator(BaseGenerator):
    """Generator for AI video script output format.

    Creates scene-by-scene scripts with:
    - Visual prompts for AI video generators
    - Voiceover narration text
    - Music suggestions per scene
    - Platform-aware timing and aspect ratios
    - Intro and outro scenes

    Required enrichments: summary
    Optional enrichments: chapters, highlights, topics
    """

    @property
    def output_type(self) -> str:
        return "ai-video-script"

    @property
    def supported_platforms(self) -> list[str]:
        return ["youtube", "tiktok", "vimeo"]

    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]

    def _get_platform_config(self, platform: str | None) -> dict:
        """Get platform configuration, defaulting to YouTube.

        Args:
            platform: Target platform name

        Returns:
            Platform configuration dictionary
        """
        platform_key = (platform or "youtube").lower()
        return PLATFORM_CONFIGS.get(platform_key, PLATFORM_CONFIGS["youtube"])

    def _calculate_scene_count(
        self,
        content_length: int,
        platform_config: dict,
    ) -> int:
        """Calculate number of scenes based on content and platform.

        Args:
            content_length: Length of source content in characters
            platform_config: Platform configuration dict

        Returns:
            Number of scenes (including intro and outro)
        """
        min_duration, max_duration = platform_config["duration_range"]
        min_scene, max_scene = platform_config["scene_duration_range"]

        # Target middle of duration range
        target_duration = (min_duration + max_duration) // 2
        avg_scene_duration = (min_scene + max_scene) // 2

        # Calculate content scenes (excluding intro/outro)
        content_scenes = max(1, target_duration // avg_scene_duration - 2)

        # Scale by content length (more content = more scenes, up to limit)
        if content_length < 500:
            content_scenes = min(content_scenes, 2)
        elif content_length < 2000:
            content_scenes = min(content_scenes, 4)

        # Add intro + outro
        return content_scenes + 2

    def _calculate_voiceover_duration(self, text: str) -> int:
        """Calculate voiceover duration in seconds from text.

        Args:
            text: Voiceover text

        Returns:
            Duration in seconds
        """
        if not text:
            return 5
        word_count = len(text.split())
        return max(5, round(word_count / WORDS_PER_MINUTE * 60))

    def _build_scenes(
        self,
        enriched_content: dict,
        platform_config: dict,
        scene_count: int,
    ) -> list[dict]:
        """Build scene list from enriched content.

        Args:
            enriched_content: The enriched content data
            platform_config: Platform configuration
            scene_count: Target number of scenes

        Returns:
            List of scene dictionaries
        """
        summary = enriched_content.get("summary", {})
        chapters = enriched_content.get("chapters", [])
        highlights = enriched_content.get("highlights", [])
        topics = enriched_content.get("topics", [])
        title = self._get_field(enriched_content, "metadata.title", "")

        min_scene_dur, max_scene_dur = platform_config["scene_duration_range"]
        avg_scene_dur = (min_scene_dur + max_scene_dur) // 2

        scenes = []

        # Scene 1: Intro
        intro_text = summary.get("short", f"Discover {title}." if title else "Welcome.")
        scenes.append({
            "scene_number": 1,
            "type": "intro",
            "visual_prompt": (
                f"Cinematic opening shot. Smooth camera movement revealing "
                f"the theme of {title or 'the topic'}. "
                f"Modern, clean aesthetic with subtle motion."
            ),
            "voiceover_text": intro_text,
            "duration_seconds": self._calculate_voiceover_duration(intro_text),
            "music_suggestion": {
                "mood": "inspiring",
                "genre": "electronic",
                "tempo": "medium",
            },
        })

        # Content scenes from chapters or summary
        content_scene_count = scene_count - 2  # minus intro and outro

        if chapters and len(chapters) > 0:
            for i, chapter in enumerate(chapters[:content_scene_count]):
                if isinstance(chapter, dict):
                    chapter_title = chapter.get("title", f"Section {i + 1}")
                    chapter_text = chapter.get("summary", "")
                else:
                    chapter_title = f"Section {i + 1}"
                    chapter_text = str(chapter)

                scenes.append({
                    "scene_number": len(scenes) + 1,
                    "type": "content",
                    "visual_prompt": (
                        f"Visual representation of {chapter_title}. "
                        f"Dynamic transitions, relevant imagery. "
                        f"Professional color grading."
                    ),
                    "voiceover_text": chapter_text,
                    "duration_seconds": self._calculate_voiceover_duration(chapter_text) or avg_scene_dur,
                    "music_suggestion": {
                        "mood": "focused",
                        "genre": "ambient",
                        "tempo": "medium",
                    },
                })
        else:
            # Build scenes from summary and highlights
            medium_summary = summary.get("medium", summary.get("long", ""))

            if medium_summary and content_scene_count >= 1:
                scenes.append({
                    "scene_number": len(scenes) + 1,
                    "type": "content",
                    "visual_prompt": (
                        f"Main content visualization. "
                        f"Engaging visuals related to {title or 'the topic'}. "
                        f"Smooth transitions between key concepts."
                    ),
                    "voiceover_text": medium_summary,
                    "duration_seconds": self._calculate_voiceover_duration(medium_summary),
                    "music_suggestion": {
                        "mood": "informative",
                        "genre": "ambient",
                        "tempo": "medium",
                    },
                })

            # Fill remaining scenes from highlights
            remaining = content_scene_count - len(scenes) + 1  # +1 for intro already added
            for i, highlight in enumerate(highlights[:remaining]):
                h_text = highlight.get("text", highlight) if isinstance(highlight, dict) else str(highlight)
                scenes.append({
                    "scene_number": len(scenes) + 1,
                    "type": "content",
                    "visual_prompt": (
                        f"Key point visualization: {h_text[:50]}. "
                        f"Bold typography overlay with supporting imagery."
                    ),
                    "voiceover_text": h_text,
                    "duration_seconds": self._calculate_voiceover_duration(h_text),
                    "music_suggestion": {
                        "mood": "engaging",
                        "genre": "electronic",
                        "tempo": "medium",
                    },
                })

        # Outro scene
        outro_text = "Thanks for watching. Like and subscribe for more content."
        scenes.append({
            "scene_number": len(scenes) + 1,
            "type": "outro",
            "visual_prompt": (
                "Closing shot with call-to-action overlay. "
                "Logo or channel branding. Fade to end card."
            ),
            "voiceover_text": outro_text,
            "duration_seconds": self._calculate_voiceover_duration(outro_text),
            "music_suggestion": {
                "mood": "uplifting",
                "genre": "electronic",
                "tempo": "medium",
            },
        })

        return scenes

    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for AI video script template.

        Args:
            enriched_content: The enriched content data
            request: The format request with options

        Returns:
            Template context dictionary
        """
        platform_config = self._get_platform_config(request.platform)
        title = self._get_field(enriched_content, "metadata.title", "")
        summary = enriched_content.get("summary", {})

        # Calculate content length for scene count
        content_text = summary.get("long", summary.get("medium", ""))
        content_length = len(content_text)

        scene_count = self._calculate_scene_count(content_length, platform_config)
        scenes = self._build_scenes(enriched_content, platform_config, scene_count)

        total_duration = sum(s["duration_seconds"] for s in scenes)

        return {
            "title": title,
            "target_platform": (request.platform or "youtube").lower(),
            "aspect_ratio": platform_config["aspect_ratio"],
            "recommended_duration_range": platform_config["duration_range"],
            "scenes": scenes,
            "scene_count": len(scenes),
            "total_duration_seconds": total_duration,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "topics": enriched_content.get("topics", []),
            "platform": request.platform,
            "tone": request.tone or "professional",
            "metadata": enriched_content.get("metadata", {}),
        }
