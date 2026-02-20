"""
IndividualEnrichment Schema

Schema for individual enrichment output files. Used when outputting
each enrichment type to a separate file (v0.8.6+ default behavior).
"""

from datetime import datetime
from typing import Any, Dict, Literal, Union
from pydantic import BaseModel, Field

from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1


class IndividualEnrichmentMetadata(BaseModel):
    """Metadata for individual enrichment file.
    
    Attributes:
        provider: LLM provider used
        model: Specific model used
        timestamp: When enrichment was performed (UTC)
        cost_usd: Cost for this enrichment type
        tokens_used: Tokens consumed for this enrichment type
    """
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Specific model used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When enrichment was performed (UTC)"
    )
    cost_usd: float = Field(..., description="Cost in USD", ge=0.0)
    tokens_used: int = Field(..., description="Tokens consumed", ge=0)


class IndividualEnrichment(BaseModel):
    """Schema for individual enrichment output file.
    
    Each separate enrichment file contains this structure with
    the enrichment type and its data.
    
    Attributes:
        enrichment_version: Schema version (always "v1")
        enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
        metadata: Metadata about this enrichment
        data: The actual enrichment data
    """
    enrichment_version: Literal["v1"] = Field(
        default="v1",
        description="Schema version"
    )
    enrichment_type: str = Field(
        ...,
        description="Type of enrichment (summary, tag, chapter, highlight)"
    )
    metadata: IndividualEnrichmentMetadata = Field(
        ...,
        description="Enrichment metadata"
    )
    data: Union[Dict[str, Any], list] = Field(
        ...,
        description="The enrichment data"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "enrichment_version": "v1",
                "enrichment_type": "summary",
                "metadata": {
                    "provider": "cloud-openai",
                    "model": "gpt-4-turbo",
                    "timestamp": "2026-02-19T10:00:00Z",
                    "cost_usd": 0.05,
                    "tokens_used": 1000
                },
                "data": {
                    "short": "A brief overview.",
                    "medium": "A paragraph summary.",
                    "long": "A detailed summary."
                }
            }
        }


def extract_individual_enrichment(
    enrichment: EnrichmentV1,
    enrichment_type: str
) -> IndividualEnrichment:
    """Extract single enrichment type from EnrichmentV1.
    
    Args:
        enrichment: Full EnrichmentV1 result
        enrichment_type: Type to extract (summary, tag, chapter, highlight)
        
    Returns:
        IndividualEnrichment with just the requested type
        
    Raises:
        ValueError: If enrichment type not found in result
    """
    # Map enrichment type to attribute name and data
    type_map = {
        "summary": ("summary", enrichment.summary),
        "tag": ("tags", enrichment.tags),
        "chapter": ("chapters", enrichment.chapters),
        "highlight": ("highlights", enrichment.highlights)
    }
    
    if enrichment_type not in type_map:
        raise ValueError(f"Unknown enrichment type: {enrichment_type}")
    
    attr_name, data = type_map[enrichment_type]
    
    if data is None:
        raise ValueError(f"Enrichment type '{enrichment_type}' not present in result")
    
    # Convert data to dict/list for serialization
    if isinstance(data, list):
        data_dict = [item.model_dump(mode='json') for item in data]
    else:
        data_dict = data.model_dump(mode='json')
    
    return IndividualEnrichment(
        enrichment_type=enrichment_type,
        metadata=IndividualEnrichmentMetadata(
            provider=enrichment.metadata.provider,
            model=enrichment.metadata.model,
            timestamp=enrichment.metadata.timestamp,
            cost_usd=enrichment.metadata.cost_usd,
            tokens_used=enrichment.metadata.tokens_used
        ),
        data=data_dict
    )
