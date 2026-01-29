"""
Enrichment Schemas

Pydantic models for enrichment data structures. These schemas define the
structure of enrichment results and ensure type safety and validation.

Import order is important to resolve forward references properly:
1. Import individual enrichment type schemas first
2. Import container schema (EnrichmentV1) last
3. Resolve forward references
"""

# Import enrichment type schemas first
from pipeline.enrichment.schemas.summary import SummaryEnrichment
from pipeline.enrichment.schemas.tag import TagEnrichment
from pipeline.enrichment.schemas.chapter import ChapterEnrichment
from pipeline.enrichment.schemas.highlight import HighlightEnrichment, ImportanceLevel

# Import container schema last
from pipeline.enrichment.schemas.enrichment_v1 import (
    EnrichmentV1,
    EnrichmentMetadata,
    _resolve_forward_refs,
)

# Resolve forward references now that all schemas are imported
_resolve_forward_refs()

__all__ = [
    "EnrichmentV1",
    "EnrichmentMetadata",
    "SummaryEnrichment",
    "TagEnrichment",
    "ChapterEnrichment",
    "HighlightEnrichment",
    "ImportanceLevel",
]
