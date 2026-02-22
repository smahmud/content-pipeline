"""
LLM Provider Factory

Factory pattern for creating LLM providers with auto-selection support.
Handles provider instantiation, caching, and intelligent provider selection
based on availability and configuration.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field

from pipeline.llm.config import LLMConfig
from pipeline.llm.providers.base import BaseLLMProvider
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.errors import ConfigurationError


@dataclass
class AutoSelectionConfig:
    """Configuration for auto-selection behavior.
    
    Attributes:
        priority_order: List of providers to try in order
        fallback_enabled: Whether to fall back to next provider if first fails
    """
    priority_order: List[str] = field(
        default_factory=lambda: ["cloud-openai", "cloud-anthropic", "cloud-aws-bedrock", "local-ollama"]
    )
    fallback_enabled: bool = True


class LLMProviderFactory:
    """Factory for creating LLM providers with auto-selection support.
    
    This factory handles:
    - Provider instantiation based on provider name
    - Provider caching to prevent redundant instantiation
    - Auto-selection of available providers
    - Provider validation before use
    
    Example:
        >>> from pipeline.llm.config import LLMConfig
        >>> llm_config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')
        >>> factory = LLMProviderFactory(llm_config)
        >>> provider = factory.create_provider("cloud-openai")
        >>> # Or use auto-selection
        >>> provider = factory.create_provider("auto")
    """
    
    def __init__(
        self,
        config: LLMConfig,
        auto_selection: Optional[AutoSelectionConfig] = None
    ):
        """Initialize LLM provider factory.
        
        Args:
            config: LLM configuration with all provider configs
            auto_selection: Configuration for auto-selection behavior
        """
        self.config = config
        self.auto_selection = auto_selection or AutoSelectionConfig()
        
        # Cache for instantiated providers
        self._provider_cache: Dict[str, BaseLLMProvider] = {}
    
    def create_provider(self, provider: str) -> BaseLLMProvider:
        """Create provider for specified provider name.
        
        Supports provider IDs: cloud-openai, cloud-anthropic, cloud-aws-bedrock, local-ollama
        
        Args:
            provider: Provider name or "auto" for auto-selection
            
        Returns:
            Instantiated LLM provider
            
        Raises:
            ConfigurationError: If provider is unknown or unavailable
        """
        # Map legacy provider names to new format
        legacy_map = {
            "openai": "cloud-openai",
            "claude": "cloud-anthropic",
            "bedrock": "cloud-aws-bedrock",
            "ollama": "local-ollama",
        }
        provider = legacy_map.get(provider, provider)

        if provider == "auto":
            return self._auto_select_provider()
        
        # Check cache first
        if provider in self._provider_cache:
            return self._provider_cache[provider]
        
        # Instantiate new provider
        provider_instance = self._instantiate_provider(provider)
        
        # Cache for future use
        self._provider_cache[provider] = provider_instance
        
        return provider_instance
    
    def _auto_select_provider(self) -> BaseLLMProvider:
        """Auto-select first available provider.
        
        Tries providers in priority order and returns the first one that
        passes validation checks.
        
        Returns:
            First available LLM provider
            
        Raises:
            ConfigurationError: If no providers are available
        """
        errors = []
        
        for provider in self.auto_selection.priority_order:
            try:
                provider_instance = self._instantiate_provider(provider)
                
                # Validate that provider is available
                if provider_instance.validate_requirements():
                    # Cache the selected provider
                    self._provider_cache[provider] = provider_instance
                    return provider_instance
                else:
                    errors.append(f"{provider}: validation failed")
                    
            except Exception as e:
                errors.append(f"{provider}: {str(e)}")
                continue
        
        # No providers available
        error_details = "\n".join(f"  - {err}" for err in errors)
        raise ConfigurationError(
            f"No LLM providers available. Tried:\n{error_details}\n\n"
            "Setup instructions:\n"
            "  - cloud-openai: Set OPENAI_API_KEY environment variable\n"
            "  - cloud-anthropic: Set ANTHROPIC_API_KEY environment variable\n"
            "  - cloud-aws-bedrock: Configure AWS credentials\n"
            "  - local-ollama: Start local service with 'ollama serve'"
        )

    
    def _instantiate_provider(self, provider: str) -> BaseLLMProvider:
        """Instantiate specific provider type.
        
        Args:
            provider: Provider name (new format: cloud-openai, cloud-anthropic, etc.)
            
        Returns:
            Instantiated provider
            
        Raises:
            ConfigurationError: If provider is unknown or not configured
        """
        if provider == "cloud-openai":
            if not self.config.openai.api_key:
                raise ConfigurationError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY environment variable or provide in config."
                )
            return CloudOpenAIProvider(self.config.openai)
        
        elif provider == "local-ollama":
            return LocalOllamaProvider(self.config.ollama)
        
        elif provider == "cloud-aws-bedrock":
            return CloudAWSBedrockProvider(self.config.bedrock)
        
        elif provider == "cloud-anthropic":
            if not self.config.anthropic.api_key:
                raise ConfigurationError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY environment variable or provide in config."
                )
            return CloudAnthropicProvider(self.config.anthropic)
        
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider}. "
                f"Valid options: cloud-openai, cloud-anthropic, cloud-aws-bedrock, local-ollama, auto"
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently available providers.
        
        Returns:
            List of provider names (new format) that pass validation
        """
        available = []
        
        for provider in ["cloud-openai", "cloud-anthropic", "cloud-aws-bedrock", "local-ollama"]:
            try:
                provider_instance = self._instantiate_provider(provider)
                if provider_instance.validate_requirements():
                    available.append(provider)
            except (ConfigurationError, NotImplementedError):
                continue
        
        return available
    
    def clear_cache(self):
        """Clear the provider cache.
        
        This forces re-instantiation of providers on next request.
        Useful for testing or when configuration changes.
        """
        self._provider_cache.clear()
