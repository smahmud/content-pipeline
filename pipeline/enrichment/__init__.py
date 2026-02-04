"""
LLM-Powered Enrichment Module

This module provides semantic enrichment of transcripts through multi-provider
LLM integration. It supports four enrichment types (Summary, Tag, Chapter, Highlight)
across four LLM providers (OpenAI, AWS Bedrock, Claude, Ollama) with comprehensive
cost control, intelligent caching, and batch processing capabilities.
"""

__version__ = "0.7.0"

# Export main components
from pipeline.enrichment.orchestrator import (
    EnrichmentOrchestrator,
    EnrichmentRequest,
    DryRunReport
)
from pipeline.enrichment.cache import CacheSystem, CacheEntry
from pipeline.enrichment.cost_estimator import CostEstimator, CostEstimate
from pipeline.llm.factory import LLMProviderFactory

__all__ = [
    "EnrichmentOrchestrator",
    "EnrichmentRequest",
    "DryRunReport",
    "CacheSystem",
    "CacheEntry",
    "CostEstimator",
    "CostEstimate",
    "LLMProviderFactory",
]
