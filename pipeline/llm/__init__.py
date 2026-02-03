"""
LLM Infrastructure Layer

This module provides shared LLM infrastructure for the Content Pipeline.
It extracts LLM provider implementations from domain modules to eliminate
cross-module dependencies and establish a clean layered architecture.

Architecture:
    Infrastructure Layer (this module)
        ↓
    Domain Layer (enrichment, formatters)
        ↓
    Application Layer (CLI)

Key Components:
    - providers/: LLM provider implementations (LocalOllamaProvider, CloudOpenAIProvider, etc.)
    - factory.py: LLMProviderFactory for provider instantiation
    - config.py: Configuration management with environment variable support
    - errors.py: LLM-specific error classes

Usage:
    >>> from pipeline.llm.factory import LLMProviderFactory
    >>> from pipeline.llm.config import LLMConfig
    >>> from pipeline.config.manager import ConfigManager
    >>> 
    >>> config_manager = ConfigManager()
    >>> llm_config = LLMConfig.load_from_config(config_manager)
    >>> factory = LLMProviderFactory(llm_config)
    >>> provider = factory.create_provider("cloud-openai")
"""

# Export main components for convenient imports
from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse

__all__ = [
    "BaseLLMProvider",
    "LLMRequest",
    "LLMResponse",
]
