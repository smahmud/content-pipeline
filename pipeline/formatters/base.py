"""
Base formatter protocol and core data classes.

Defines the BaseFormatter protocol that all output type generators
must implement, along with FormatRequest and FormatResult dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pipeline.formatters.schemas.format_v1 import FormatV1


class OutputType(str, Enum):
    """Supported output types for formatting."""
    
    # Tier 1 - Core formats
    BLOG = "blog"
    TWEET = "tweet"
    YOUTUBE = "youtube"
    SEO = "seo"
    
    # Tier 2 - Extended formats
    LINKEDIN = "linkedin"
    NEWSLETTER = "newsletter"
    CHAPTERS = "chapters"
    TRANSCRIPT_CLEAN = "transcript-clean"
    
    # Tier 3 - Specialized formats
    PODCAST_NOTES = "podcast-notes"
    MEETING_MINUTES = "meeting-minutes"
    SLIDES = "slides"
    NOTION = "notion"
    OBSIDIAN = "obsidian"
    QUOTE_CARDS = "quote-cards"
    
    # AI Video formats
    VIDEO_SCRIPT = "video-script"
    TIKTOK_SCRIPT = "tiktok-script"


class Platform(str, Enum):
    """Supported publishing platforms."""
    
    # Blog platforms
    MEDIUM = "medium"
    WORDPRESS = "wordpress"
    GHOST = "ghost"
    SUBSTACK = "substack"
    
    # Social platforms
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"


class Tone(str, Enum):
    """Available tone options for content generation."""
    
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"


class Length(str, Enum):
    """Available length options for content generation."""
    
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


# All valid output types as a list for validation
VALID_OUTPUT_TYPES = [t.value for t in OutputType]

# All valid platforms as a list for validation
VALID_PLATFORMS = [p.value for p in Platform]

# All valid tones as a list for validation
VALID_TONES = [t.value for t in Tone]

# All valid lengths as a list for validation
VALID_LENGTHS = [l.value for l in Length]


@dataclass
class FormatRequest:
    """Request for formatting operation.
    
    Attributes:
        enriched_content: EnrichmentV1 data dictionary
        output_type: Target output format (e.g., "blog", "tweet")
        platform: Optional target platform (e.g., "medium", "twitter")
        style_profile: Optional parsed style profile dictionary
        tone: Optional tone override (professional, casual, technical, friendly)
        length: Optional length override (short, medium, long)
        llm_enhance: Whether to enable LLM enhancement (default True)
        provider: LLM provider to use (default "auto")
        model: Optional specific model to use
        max_cost: Optional maximum cost limit in USD
        dry_run: If True, estimate cost without execution
        url: Optional URL to include in promotional content (e.g., link to blog/linkedin post)
    """
    
    enriched_content: dict
    output_type: str
    platform: Optional[str] = None
    style_profile: Optional[dict] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    llm_enhance: bool = True
    provider: str = "auto"
    model: Optional[str] = None
    max_cost: Optional[float] = None
    dry_run: bool = False
    url: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if self.output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                f"Invalid output type: {self.output_type}. "
                f"Valid types: {VALID_OUTPUT_TYPES}"
            )
        
        if self.platform is not None and self.platform not in VALID_PLATFORMS:
            raise ValueError(
                f"Invalid platform: {self.platform}. "
                f"Valid platforms: {VALID_PLATFORMS}"
            )
        
        if self.tone is not None and self.tone not in VALID_TONES:
            raise ValueError(
                f"Invalid tone: {self.tone}. "
                f"Valid tones: {VALID_TONES}"
            )
        
        if self.length is not None and self.length not in VALID_LENGTHS:
            raise ValueError(
                f"Invalid length: {self.length}. "
                f"Valid lengths: {VALID_LENGTHS}"
            )


@dataclass
class FormatResult:
    """Result of formatting operation.
    
    Attributes:
        content: The formatted content string
        metadata: FormatV1 metadata object
        warnings: List of any warnings (truncation, fallback, etc.)
        success: Whether the operation succeeded
        error: Optional error message if operation failed
    """
    
    content: str
    metadata: "FormatV1"
    warnings: list[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class CostEstimate:
    """Cost estimate for LLM enhancement.
    
    Attributes:
        estimated_tokens: Estimated token count
        estimated_cost_usd: Estimated cost in USD
        provider: LLM provider for estimate
        model: Model used for estimate
        within_budget: Whether estimate is within max_cost limit
    """
    
    estimated_tokens: int
    estimated_cost_usd: float
    provider: str
    model: str
    within_budget: bool = True


@runtime_checkable
class BaseFormatter(Protocol):
    """Protocol for all output type generators.
    
    Each output type (blog, tweet, youtube, etc.) must implement
    this protocol to be used by the FormatComposer.
    """
    
    @property
    def output_type(self) -> str:
        """Return the output type this formatter handles.
        
        Returns:
            Output type string (e.g., "blog", "tweet")
        """
        ...
    
    @property
    def supported_platforms(self) -> list[str]:
        """Return list of supported platforms for this output type.
        
        Returns:
            List of platform strings (e.g., ["medium", "wordpress"])
        """
        ...
    
    @property
    def required_enrichments(self) -> list[str]:
        """Return list of required enrichment types.
        
        Returns:
            List of required enrichment fields (e.g., ["summary", "tags"])
        """
        ...
    
    def format(self, request: FormatRequest) -> FormatResult:
        """Generate formatted output from enriched content.
        
        Args:
            request: FormatRequest with enriched content and options
            
        Returns:
            FormatResult with formatted content and metadata
        """
        ...
    
    def validate_input(self, enriched_content: dict) -> tuple[bool, list[str]]:
        """Validate that enriched content has required fields.
        
        Args:
            enriched_content: EnrichmentV1 data dictionary
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        ...
