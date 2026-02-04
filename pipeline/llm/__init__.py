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
    >>> from pipeline.llm import LLMProviderFactory, LLMConfig
    >>> 
    >>> # Load configuration from YAML
    >>> llm_config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')
    >>> 
    >>> # Create factory and instantiate provider
    >>> factory = LLMProviderFactory(llm_config)
    >>> provider = factory.create_provider("cloud-openai")
    >>> 
    >>> # Make a request
    >>> from pipeline.llm import LLMRequest
    >>> request = LLMRequest(prompt="Hello", max_tokens=100, temperature=0.7)
    >>> response = provider.generate(request)
"""

# Export base classes and data structures
from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse

# Export factory
from pipeline.llm.factory import LLMProviderFactory, AutoSelectionConfig

# Export configuration classes
from pipeline.llm.config import (
    LLMConfig,
    OllamaConfig,
    OpenAIConfig,
    BedrockConfig,
    AnthropicConfig,
)

# Export error classes
from pipeline.llm.errors import (
    LLMError,
    ConfigurationError,
    ProviderError,
    ProviderNotAvailableError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)

__all__ = [
    # Base classes and data structures
    "BaseLLMProvider",
    "LLMRequest",
    "LLMResponse",
    
    # Factory
    "LLMProviderFactory",
    "AutoSelectionConfig",
    
    # Configuration
    "LLMConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "BedrockConfig",
    "AnthropicConfig",
    
    # Errors
    "LLMError",
    "ConfigurationError",
    "ProviderError",
    "ProviderNotAvailableError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
]
