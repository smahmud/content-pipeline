"""
FormatV1 Pydantic schema for formatted output metadata.

This schema defines the metadata structure that accompanies
all formatted outputs, tracking provenance, LLM usage, and
validation results.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer


class LLMMetadata(BaseModel):
    """LLM enhancement metadata.
    
    Tracks details about LLM usage when enhancement is enabled.
    """
    
    provider: str = Field(
        ...,
        description="LLM provider used (openai, claude, bedrock)"
    )
    model: str = Field(
        ...,
        description="Specific model used for enhancement"
    )
    cost_usd: float = Field(
        ...,
        ge=0,
        description="Cost of LLM enhancement in USD"
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed (input + output)"
    )
    temperature: float = Field(
        ...,
        ge=0,
        le=2,
        description="Temperature setting used"
    )
    enhanced: bool = Field(
        default=True,
        description="Whether LLM enhancement was applied"
    )


class ValidationMetadata(BaseModel):
    """Platform validation metadata.
    
    Tracks validation results and any content modifications.
    """
    
    platform: Optional[str] = Field(
        default=None,
        description="Target platform for validation"
    )
    character_count: int = Field(
        ...,
        ge=0,
        description="Final character count of content"
    )
    truncated: bool = Field(
        default=False,
        description="Whether content was truncated to fit limits"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings (truncation, limit approaching, etc.)"
    )


class FormatV1(BaseModel):
    """Schema for formatted output metadata.
    
    This metadata accompanies all formatted outputs and provides
    complete provenance tracking for the formatting operation.
    
    Attributes:
        format_version: Schema version (always "v1")
        output_type: Type of output generated (blog, tweet, etc.)
        platform: Target platform if specified
        timestamp: When the formatting was performed
        source_file: Path to the input enriched JSON file
        style_profile_used: Name of style profile if used
        llm_metadata: LLM enhancement details if enhancement was used
        validation: Platform validation results
        tone: Tone setting used (if specified)
        length: Length setting used (if specified)
    """
    
    format_version: str = Field(
        default="v1",
        description="Schema version identifier"
    )
    output_type: str = Field(
        ...,
        description="Output type generated (blog, tweet, youtube, etc.)"
    )
    platform: Optional[str] = Field(
        default=None,
        description="Target platform for the output"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of formatting operation"
    )
    source_file: str = Field(
        ...,
        description="Path to the source enriched JSON file"
    )
    style_profile_used: Optional[str] = Field(
        default=None,
        description="Name of style profile used for formatting"
    )
    
    # LLM metadata (present when enhancement is used)
    llm_metadata: Optional[LLMMetadata] = Field(
        default=None,
        description="LLM enhancement metadata (if enhancement was used)"
    )
    
    # Validation metadata (always present)
    validation: ValidationMetadata = Field(
        ...,
        description="Platform validation results"
    )
    
    # Generation settings
    tone: Optional[str] = Field(
        default=None,
        description="Tone setting used (professional, casual, technical, friendly)"
    )
    length: Optional[str] = Field(
        default=None,
        description="Length setting used (short, medium, long)"
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return timestamp.isoformat()
    
    def to_frontmatter(self) -> str:
        """Convert metadata to YAML frontmatter format.
        
        Returns:
            YAML frontmatter string for embedding in Markdown outputs
        """
        lines = [
            "---",
            f"format_version: {self.format_version}",
            f"output_type: {self.output_type}",
        ]
        
        if self.platform:
            lines.append(f"platform: {self.platform}")
        
        lines.append(f"timestamp: {self.timestamp.isoformat()}")
        lines.append(f"source_file: {self.source_file}")
        
        if self.style_profile_used:
            lines.append(f"style_profile: {self.style_profile_used}")
        
        if self.tone:
            lines.append(f"tone: {self.tone}")
        
        if self.length:
            lines.append(f"length: {self.length}")
        
        # Validation section
        lines.append("validation:")
        lines.append(f"  character_count: {self.validation.character_count}")
        lines.append(f"  truncated: {str(self.validation.truncated).lower()}")
        if self.validation.warnings:
            lines.append("  warnings:")
            for warning in self.validation.warnings:
                lines.append(f"    - {warning}")
        
        # LLM metadata section (if present)
        if self.llm_metadata:
            lines.append("llm:")
            lines.append(f"  provider: {self.llm_metadata.provider}")
            lines.append(f"  model: {self.llm_metadata.model}")
            lines.append(f"  cost_usd: {self.llm_metadata.cost_usd:.6f}")
            lines.append(f"  tokens_used: {self.llm_metadata.tokens_used}")
            lines.append(f"  temperature: {self.llm_metadata.temperature}")
            lines.append(f"  enhanced: {str(self.llm_metadata.enhanced).lower()}")
        
        lines.append("---")
        return "\n".join(lines)
