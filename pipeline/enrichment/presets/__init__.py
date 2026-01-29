"""
Enrichment Presets

This module provides preset configurations for enrichment operations,
including quality presets (fast, balanced, best) and content profiles
(podcast, meeting, lecture).
"""

from pipeline.enrichment.presets.quality import (
    QualityPreset,
    QualityPresets,
    get_quality_preset
)
from pipeline.enrichment.presets.content import (
    ContentProfile,
    ContentProfiles,
    apply_content_profile,
    merge_profile_with_cli_flags
)

__all__ = [
    "QualityPreset",
    "QualityPresets",
    "get_quality_preset",
    "ContentProfile",
    "ContentProfiles",
    "apply_content_profile",
    "merge_profile_with_cli_flags",
]
