"""
Platform validator for enforcing platform-specific constraints.

This module provides validation and truncation functionality to ensure
formatted content meets platform character limits and formatting rules.

Implements:
- Property 3: Platform Character Limit Enforcement
- Property 4: Intelligent Truncation at Sentence Boundaries
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TruncationWarning(str, Enum):
    """Standard truncation warning messages."""
    
    TRUNCATED = "Content truncated to fit platform character limit"
    SENTENCE_BOUNDARY = "Content truncated at sentence boundary"
    NO_SENTENCE_BOUNDARY = "Content truncated mid-sentence (no sentence boundary found)"


@dataclass
class PlatformLimits:
    """Platform-specific constraints.
    
    Attributes:
        max_chars: Maximum character count (None = no limit)
        max_hashtags: Maximum hashtag count (None = no limit)
        max_links: Maximum link count (None = no limit)
        allowed_formatting: List of allowed formatting types
    """
    
    max_chars: Optional[int] = None
    max_hashtags: Optional[int] = None
    max_links: Optional[int] = None
    allowed_formatting: list[str] = field(default_factory=lambda: ["bold", "italic", "links"])


@dataclass
class ValidationResult:
    """Result of platform validation.
    
    Attributes:
        is_valid: Whether content meets platform constraints
        character_count: Actual character count
        exceeds_limit: Whether content exceeds character limit
        warnings: List of validation warnings
        truncated: Whether content was truncated
    """
    
    is_valid: bool
    character_count: int
    exceeds_limit: bool = False
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False


class PlatformValidator:
    """Validates and enforces platform-specific constraints.
    
    This class provides methods to:
    - Validate content against platform character limits
    - Truncate content intelligently at sentence boundaries
    - Get platform-specific limits
    
    Implements Property 3 (Platform Character Limit Enforcement) and
    Property 4 (Intelligent Truncation at Sentence Boundaries).
    """
    
    # Platform limits as defined in design document
    PLATFORM_LIMITS: dict[str, PlatformLimits] = {
        # Social platforms
        "twitter": PlatformLimits(
            max_chars=280,
            max_hashtags=5,
            max_links=1,
            allowed_formatting=["links"],
        ),
        "linkedin": PlatformLimits(
            max_chars=3000,
            max_hashtags=10,
            max_links=None,
            allowed_formatting=["bold", "italic", "links", "lists"],
        ),
        "youtube": PlatformLimits(
            max_chars=5000,
            max_hashtags=15,
            max_links=None,
            allowed_formatting=["links"],
        ),
        
        # Blog platforms (no character limits)
        "medium": PlatformLimits(
            max_chars=None,
            allowed_formatting=["bold", "italic", "links", "headers", "code", "lists", "images"],
        ),
        "wordpress": PlatformLimits(
            max_chars=None,
            allowed_formatting=["bold", "italic", "links", "headers", "code", "lists", "images"],
        ),
        "ghost": PlatformLimits(
            max_chars=None,
            allowed_formatting=["bold", "italic", "links", "headers", "code", "lists", "images"],
        ),
        "substack": PlatformLimits(
            max_chars=None,
            allowed_formatting=["bold", "italic", "links", "headers", "code", "lists", "images"],
        ),
        
        # SEO-specific limits
        "meta_title": PlatformLimits(
            max_chars=60,
            allowed_formatting=[],
        ),
        "meta_description": PlatformLimits(
            max_chars=160,
            allowed_formatting=[],
        ),
        
        # Video platforms
        "tiktok": PlatformLimits(
            max_chars=None,
            allowed_formatting=["links"],
        ),
        "vimeo": PlatformLimits(
            max_chars=None,
            allowed_formatting=["links"],
        ),
    }
    
    # Sentence ending characters
    SENTENCE_ENDINGS = frozenset(".!?")
    
    def __init__(self) -> None:
        """Initialize the platform validator."""
        pass
    
    def get_limits(self, platform: str) -> PlatformLimits:
        """Get limits for a specific platform.
        
        Args:
            platform: Platform identifier (e.g., "twitter", "linkedin")
            
        Returns:
            PlatformLimits for the specified platform
            
        Raises:
            ValueError: If platform is not recognized
        """
        platform_lower = platform.lower()
        if platform_lower not in self.PLATFORM_LIMITS:
            raise ValueError(
                f"Unknown platform: {platform}. "
                f"Valid platforms: {list(self.PLATFORM_LIMITS.keys())}"
            )
        return self.PLATFORM_LIMITS[platform_lower]
    
    def validate(self, content: str, platform: str) -> ValidationResult:
        """Validate content against platform limits.
        
        Implements Property 3: Platform Character Limit Enforcement.
        
        Args:
            content: Content string to validate
            platform: Target platform identifier
            
        Returns:
            ValidationResult with validation status and details
        """
        limits = self.get_limits(platform)
        char_count = len(content)
        warnings: list[str] = []
        exceeds_limit = False
        
        # Check character limit
        if limits.max_chars is not None and char_count > limits.max_chars:
            exceeds_limit = True
            warnings.append(
                f"Content exceeds {platform} character limit: "
                f"{char_count} > {limits.max_chars}"
            )
        
        # Check hashtag limit
        if limits.max_hashtags is not None:
            hashtag_count = content.count("#")
            if hashtag_count > limits.max_hashtags:
                warnings.append(
                    f"Content exceeds {platform} hashtag limit: "
                    f"{hashtag_count} > {limits.max_hashtags}"
                )
        
        # Check link limit (simple URL detection)
        if limits.max_links is not None:
            # Count http:// and https:// occurrences
            link_count = content.lower().count("http://") + content.lower().count("https://")
            if link_count > limits.max_links:
                warnings.append(
                    f"Content exceeds {platform} link limit: "
                    f"{link_count} > {limits.max_links}"
                )
        
        return ValidationResult(
            is_valid=not exceeds_limit,
            character_count=char_count,
            exceeds_limit=exceeds_limit,
            warnings=warnings,
            truncated=False,
        )
    
    def truncate(
        self,
        content: str,
        platform: str,
        ellipsis: str = "...",
    ) -> tuple[str, bool, list[str]]:
        """Truncate content to fit platform limits at sentence boundaries.
        
        Implements Property 4: Intelligent Truncation at Sentence Boundaries.
        
        This method attempts to truncate content at a sentence boundary
        (ending with '.', '!', or '?') to preserve readability. If no
        sentence boundary is found within the limit, it truncates at the
        limit and adds an ellipsis.
        
        Args:
            content: Content string to truncate
            platform: Target platform identifier
            ellipsis: String to append when truncating (default "...")
            
        Returns:
            Tuple of (truncated_content, was_truncated, warnings)
        """
        limits = self.get_limits(platform)
        warnings: list[str] = []
        
        # If no character limit or content fits, return as-is
        if limits.max_chars is None or len(content) <= limits.max_chars:
            return content, False, warnings
        
        # Calculate max length accounting for ellipsis
        max_length = limits.max_chars - len(ellipsis)
        
        if max_length <= 0:
            # Edge case: limit is smaller than ellipsis
            return content[:limits.max_chars], True, [TruncationWarning.TRUNCATED.value]
        
        # Find the last sentence boundary within the limit
        truncation_point = self._find_sentence_boundary(content, max_length)
        
        if truncation_point > 0:
            # Found a sentence boundary - truncate there
            truncated = content[:truncation_point].rstrip()
            warnings.append(TruncationWarning.TRUNCATED.value)
            warnings.append(TruncationWarning.SENTENCE_BOUNDARY.value)
            return truncated, True, warnings
        else:
            # No sentence boundary found - truncate at limit with ellipsis
            truncated = content[:max_length].rstrip() + ellipsis
            warnings.append(TruncationWarning.TRUNCATED.value)
            warnings.append(TruncationWarning.NO_SENTENCE_BOUNDARY.value)
            return truncated, True, warnings
    
    def _find_sentence_boundary(self, content: str, max_length: int) -> int:
        """Find the last sentence boundary within the given length.
        
        Args:
            content: Content string to search
            max_length: Maximum position to search up to
            
        Returns:
            Position after the sentence ending character, or 0 if not found
        """
        # Search backwards from max_length for sentence endings
        search_text = content[:max_length]
        
        # Find the last occurrence of each sentence ending
        last_boundary = 0
        
        for i in range(len(search_text) - 1, -1, -1):
            if search_text[i] in self.SENTENCE_ENDINGS:
                # Check if this is followed by whitespace or end of search area
                # to avoid matching abbreviations like "Dr." in the middle of text
                if i == len(search_text) - 1:
                    # At the end of search area
                    last_boundary = i + 1
                    break
                elif search_text[i + 1].isspace():
                    # Followed by whitespace - likely a real sentence end
                    last_boundary = i + 1
                    break
        
        return last_boundary
    
    def validate_and_truncate(
        self,
        content: str,
        platform: str,
        auto_truncate: bool = True,
    ) -> tuple[str, ValidationResult]:
        """Validate content and optionally truncate to fit limits.
        
        Convenience method that combines validation and truncation.
        
        Args:
            content: Content string to validate/truncate
            platform: Target platform identifier
            auto_truncate: If True, automatically truncate content that exceeds limits
            
        Returns:
            Tuple of (final_content, ValidationResult)
        """
        # First validate
        result = self.validate(content, platform)
        
        if not result.exceeds_limit or not auto_truncate:
            return content, result
        
        # Truncate if needed
        truncated_content, was_truncated, truncate_warnings = self.truncate(
            content, platform
        )
        
        # Re-validate truncated content
        final_result = self.validate(truncated_content, platform)
        final_result.truncated = was_truncated
        final_result.warnings.extend(truncate_warnings)
        
        return truncated_content, final_result


class PlatformValidatorError(Exception):
    """Exception raised for platform validation errors."""
    
    pass
