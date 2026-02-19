"""
LLM Enhancement Module

Provides LLM-powered prose enhancement for formatted content.
Reuses the agent infrastructure from the enrichment module.
"""

from pipeline.formatters.llm.enhancer import (
    LLMEnhancer,
    EnhancementResult,
    EnhancementConfig,
)

__all__ = [
    "LLMEnhancer",
    "EnhancementResult",
    "EnhancementConfig",
]
