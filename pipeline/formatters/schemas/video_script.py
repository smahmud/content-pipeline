"""
AI Video Script schemas for video generation tools.

Defines data models for scene-by-scene video scripts compatible with
AI video generation tools like Sora, Runway, and Pika.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MusicSuggestion:
    """Background music recommendation for a scene.
    
    Attributes:
        mood: Emotional tone (e.g., "upbeat", "calm", "dramatic")
        genre: Music style (e.g., "electronic", "acoustic", "orchestral")
        tempo: Speed indicator (e.g., "slow", "medium", "fast")
    """
    mood: str
    genre: str
    tempo: Literal["slow", "medium", "fast"]


@dataclass
class VideoScene:
    """A single scene in the AI video script.
    
    Attributes:
        scene_number: Sequential scene identifier (1-based)
        duration_seconds: Length of scene in seconds
        visual_prompt: Detailed description for AI video generator
        voiceover_text: Narration text for the scene
        music_suggestion: Background music recommendation
    """
    scene_number: int
    duration_seconds: int
    visual_prompt: str
    voiceover_text: str
    music_suggestion: MusicSuggestion


@dataclass
class VideoMetadata:
    """Metadata for the AI video script.
    
    Attributes:
        title: Video title
        target_platform: Destination platform (youtube, tiktok, vimeo)
        aspect_ratio: Video dimensions ratio (e.g., "16:9", "9:16")
        recommended_duration_range: Min and max duration in seconds
    """
    title: str
    target_platform: Literal["youtube", "tiktok", "vimeo"]
    aspect_ratio: str
    recommended_duration_range: tuple[int, int]


@dataclass
class AIVideoScript:
    """Complete AI video script with scenes and metadata.
    
    Attributes:
        schema_version: Schema version identifier
        metadata: Video metadata including platform and duration
        scenes: List of video scenes
        total_duration_seconds: Sum of all scene durations
    """
    schema_version: str = "ai_video_script_v1"
    metadata: Optional[VideoMetadata] = None
    scenes: list[VideoScene] = field(default_factory=list)
    total_duration_seconds: int = 0
    
    def __post_init__(self) -> None:
        """Calculate total duration from scenes."""
        if self.scenes and self.total_duration_seconds == 0:
            self.total_duration_seconds = sum(s.duration_seconds for s in self.scenes)


# Platform-specific configurations
PLATFORM_CONFIGS = {
    "youtube": {
        "aspect_ratio": "16:9",
        "duration_range": (180, 600),  # 3-10 minutes
        "scene_duration_range": (10, 30),
    },
    "tiktok": {
        "aspect_ratio": "9:16",
        "duration_range": (15, 60),  # 15-60 seconds
        "scene_duration_range": (3, 10),
    },
    "vimeo": {
        "aspect_ratio": "16:9",
        "duration_range": (60, 1800),  # 1-30 minutes
        "scene_duration_range": (15, 45),
    },
}

# Words per minute for voiceover timing calculation
WORDS_PER_MINUTE = 150
