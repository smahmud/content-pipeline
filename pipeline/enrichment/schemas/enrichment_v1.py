"""
EnrichmentV1 Container Schema

Main container schema for all enrichment results. This schema aggregates
all enrichment types (summary, tags, chapters, highlights) along with
metadata about the enrichment operation.
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING, Literal
from pydantic import BaseModel, Field

# Import enrichment type schemas
if TYPE_CHECKING:
    # Use TYPE_CHECKING to avoid circular imports during type checking
    from pipeline.enrichment.schemas.summary import SummaryEnrichment
    from pipeline.enrichment.schemas.tag import TagEnrichment
    from pipeline.enrichment.schemas.chapter import ChapterEnrichment
    from pipeline.enrichment.schemas.highlight import HighlightEnrichment


class EnrichmentMetadata(BaseModel):
    """Metadata about the enrichment operation.
    
    This captures information about how the enrichment was performed,
    including the LLM provider, model, cost, and timing information.
    
    Attributes:
        provider: LLM provider used (openai, bedrock, claude, ollama)
        model: Specific model used (e.g., gpt-4-turbo, llama2)
        timestamp: When the enrichment was performed (UTC)
        cost_usd: Total cost in USD (0.0 for local models)
        tokens_used: Total tokens consumed (input + output)
        enrichment_types: List of enrichment types performed
        cache_hit: Whether result was retrieved from cache
    """
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Specific model used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When enrichment was performed (UTC)"
    )
    cost_usd: float = Field(..., description="Total cost in USD", ge=0.0)
    tokens_used: int = Field(..., description="Total tokens consumed", ge=0)
    enrichment_types: List[str] = Field(
        ...,
        description="Types of enrichment performed",
        min_items=1
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether result was from cache"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-4-turbo",
                "timestamp": "2026-01-29T10:30:00Z",
                "cost_usd": 0.31,
                "tokens_used": 8300,
                "enrichment_types": ["summary", "tags", "chapters", "highlights"],
                "cache_hit": False
            }
        }


class EnrichmentV1(BaseModel):
    """Container for all enrichment results.
    
    This is the top-level schema that contains all enrichment data.
    It includes metadata about the operation and optional fields for
    each enrichment type (summary, tags, chapters, highlights).
    
    The schema is versioned (v1) to allow for future evolution while
    maintaining backward compatibility.
    
    Attributes:
        enrichment_version: Schema version (always "v1")
        metadata: Metadata about the enrichment operation
        summary: Optional summary enrichment (short, medium, long)
        tags: Optional tag enrichment (categories, keywords, entities)
        chapters: Optional list of chapter enrichments
        highlights: Optional list of highlight enrichments
    """
    enrichment_version: Literal["v1"] = Field(
        default="v1",
        description="Schema version"
    )
    metadata: EnrichmentMetadata = Field(..., description="Enrichment metadata")
    
    # Optional enrichment types - at least one must be present
    summary: Optional["SummaryEnrichment"] = Field(
        None,
        description="Summary enrichment with short, medium, and long variants"
    )
    tags: Optional["TagEnrichment"] = Field(
        None,
        description="Tag enrichment with categories, keywords, and entities"
    )
    chapters: Optional[List["ChapterEnrichment"]] = Field(
        None,
        description="List of chapter enrichments with timestamps"
    )
    highlights: Optional[List["HighlightEnrichment"]] = Field(
        None,
        description="List of highlight enrichments with timestamps"
    )
    
    def model_post_init(self, __context):
        """Validate that at least one enrichment type is present."""
        if not any([self.summary, self.tags, self.chapters, self.highlights]):
            raise ValueError(
                "At least one enrichment type must be present "
                "(summary, tags, chapters, or highlights)"
            )
    
    class Config:
        json_schema_extra = {
            "example": {
                "enrichment_version": "v1",
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4-turbo",
                    "timestamp": "2026-01-29T10:30:00Z",
                    "cost_usd": 0.31,
                    "tokens_used": 8300,
                    "enrichment_types": ["summary", "tags"],
                    "cache_hit": False
                },
                "summary": {
                    "short": "A brief overview of the content.",
                    "medium": "A paragraph-length summary with key points.",
                    "long": "A detailed multi-paragraph summary."
                },
                "tags": {
                    "categories": ["Technology", "AI"],
                    "keywords": ["machine learning", "neural networks"],
                    "entities": ["OpenAI", "GPT-4"]
                }
            }
        }


# Resolve forward references after all schemas are defined
# This is called when the module is imported to ensure proper type resolution
def _resolve_forward_refs():
    """Resolve forward references in EnrichmentV1 model."""
    try:
        # Import actual schemas (not just for type checking)
        from pipeline.enrichment.schemas.summary import SummaryEnrichment
        from pipeline.enrichment.schemas.tag import TagEnrichment
        from pipeline.enrichment.schemas.chapter import ChapterEnrichment
        from pipeline.enrichment.schemas.highlight import HighlightEnrichment
        
        # Rebuild model with resolved references
        EnrichmentV1.model_rebuild()
    except ImportError:
        # Schemas not yet available, will be resolved when they are imported
        pass
