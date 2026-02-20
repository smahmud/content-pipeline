"""
Pydantic schemas for formatter data models.

Contains FormatV1 schema for output metadata and validation,
and AIVideoScript schemas for video generation.
"""

from pipeline.formatters.schemas.format_v1 import (
    FormatV1,
    LLMMetadata,
    ValidationMetadata,
)
from pipeline.formatters.schemas.video_script import (
    AIVideoScript,
    MusicSuggestion,
    PLATFORM_CONFIGS,
    VideoMetadata,
    VideoScene,
    WORDS_PER_MINUTE,
)

__all__ = [
    "FormatV1",
    "LLMMetadata",
    "ValidationMetadata",
    "AIVideoScript",
    "MusicSuggestion",
    "PLATFORM_CONFIGS",
    "VideoMetadata",
    "VideoScene",
    "WORDS_PER_MINUTE",
]
