"""
Content Profiles

Defines domain-specific enrichment configurations for different
content types (podcast, meeting, lecture). Each profile specifies
which enrichment types to enable and how to configure them.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class ContentProfile:
    """Content profile configuration.
    
    Attributes:
        name: Profile name (podcast, meeting, lecture, custom)
        description: Human-readable description
        enrichment_types: List of enrichment types to enable
        summary_length: Preferred summary length (short, medium, long)
        extract_speakers: Whether to extract speaker information
        detect_chapters: Whether to detect chapters
        highlight_importance: Minimum importance level for highlights
        custom_instructions: Optional custom instructions for prompts
    """
    name: str
    description: str
    enrichment_types: List[str]
    summary_length: str = "medium"
    extract_speakers: bool = False
    detect_chapters: bool = True
    highlight_importance: str = "medium"
    custom_instructions: Optional[Dict[str, str]] = None


class ContentProfiles:
    """Content profile definitions for different content types.
    
    This class defines profiles for:
    - PODCAST: Long-form audio content with chapters and highlights
    - MEETING: Business meetings with action items and decisions
    - LECTURE: Educational content with key concepts and structure
    
    Example:
        >>> profile = ContentProfiles.PODCAST
        >>> enrichment_types = profile.enrichment_types
        >>> # Returns ["summary", "tag", "chapter", "highlight"]
    """
    
    PODCAST = ContentProfile(
        name="podcast",
        description="Podcast or long-form audio content",
        enrichment_types=["summary", "tag", "chapter", "highlight"],
        summary_length="medium",
        extract_speakers=True,
        detect_chapters=True,
        highlight_importance="medium",
        custom_instructions={
            "summary": "Focus on main topics and key takeaways from the conversation.",
            "tag": "Extract speaker names, topics discussed, and key entities mentioned.",
            "chapter": "Identify natural topic transitions and conversation segments.",
            "highlight": "Mark memorable quotes, insights, and important moments."
        }
    )
    
    MEETING = ContentProfile(
        name="meeting",
        description="Business meeting or conference call",
        enrichment_types=["summary", "tag", "highlight"],
        summary_length="short",
        extract_speakers=True,
        detect_chapters=False,  # Meetings typically don't have clear chapters
        highlight_importance="high",  # Only highlight important decisions/actions
        custom_instructions={
            "summary": "Summarize key decisions, action items, and outcomes.",
            "tag": "Extract attendees, topics, action items, and decisions.",
            "highlight": "Identify action items, decisions, and important commitments."
        }
    )
    
    LECTURE = ContentProfile(
        name="lecture",
        description="Educational lecture or presentation",
        enrichment_types=["summary", "tag", "chapter"],
        summary_length="long",
        extract_speakers=False,  # Usually single speaker
        detect_chapters=True,
        highlight_importance="medium",
        custom_instructions={
            "summary": "Provide comprehensive overview of concepts and learning objectives.",
            "tag": "Extract key concepts, terminology, and subject areas.",
            "chapter": "Identify major topics and subtopics in the lecture structure."
        }
    )
    
    @classmethod
    def get_profile(cls, name: str) -> ContentProfile:
        """Get profile by name.
        
        Args:
            name: Profile name (podcast, meeting, lecture)
            
        Returns:
            Content profile
            
        Raises:
            ValueError: If profile name is invalid
        """
        profiles = {
            "podcast": cls.PODCAST,
            "meeting": cls.MEETING,
            "lecture": cls.LECTURE
        }
        
        if name not in profiles:
            raise ValueError(
                f"Invalid content profile: {name}. "
                f"Valid options: {', '.join(profiles.keys())}"
            )
        
        return profiles[name]
    
    @classmethod
    def list_profiles(cls) -> Dict[str, str]:
        """List all available profiles with descriptions.
        
        Returns:
            Dict mapping profile name to description
        """
        return {
            "podcast": cls.PODCAST.description,
            "meeting": cls.MEETING.description,
            "lecture": cls.LECTURE.description
        }
    
    @classmethod
    def create_custom_profile(
        cls,
        name: str,
        description: str,
        enrichment_types: List[str],
        **kwargs
    ) -> ContentProfile:
        """Create a custom content profile.
        
        Args:
            name: Profile name
            description: Profile description
            enrichment_types: List of enrichment types to enable
            **kwargs: Additional profile parameters
            
        Returns:
            Custom content profile
        """
        return ContentProfile(
            name=name,
            description=description,
            enrichment_types=enrichment_types,
            **kwargs
        )


def apply_content_profile(
    profile: ContentProfile,
    base_enrichment_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Apply content profile to get enrichment configuration.
    
    Args:
        profile: Content profile to apply
        base_enrichment_types: Optional base enrichment types (profile types take precedence)
        
    Returns:
        Dict with enrichment configuration
    """
    # Use profile enrichment types if specified, otherwise use base
    enrichment_types = profile.enrichment_types or base_enrichment_types or []
    
    return {
        "enrichment_types": enrichment_types,
        "summary_length": profile.summary_length,
        "extract_speakers": profile.extract_speakers,
        "detect_chapters": profile.detect_chapters,
        "highlight_importance": profile.highlight_importance,
        "custom_instructions": profile.custom_instructions or {}
    }


def merge_profile_with_cli_flags(
    profile: ContentProfile,
    cli_enrichment_types: Optional[List[str]] = None
) -> List[str]:
    """Merge content profile with CLI flags.
    
    CLI flags take precedence over profile settings.
    
    Args:
        profile: Content profile
        cli_enrichment_types: Enrichment types from CLI flags
        
    Returns:
        Merged list of enrichment types
    """
    # If CLI flags are specified, use them
    if cli_enrichment_types:
        return cli_enrichment_types
    
    # Otherwise use profile defaults
    return profile.enrichment_types
