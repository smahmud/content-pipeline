"""
LLM Agent Factory

Factory pattern for creating LLM agents with auto-selection support.
Handles agent instantiation, caching, and intelligent provider selection
based on availability and configuration.
"""

import os
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from pipeline.enrichment.agents.base import BaseLLMAgent
from pipeline.enrichment.agents.cloud_openai_agent import CloudOpenAIAgent, CloudOpenAIAgentConfig
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent, LocalOllamaAgentConfig
from pipeline.enrichment.agents.cloud_aws_bedrock_agent import CloudAWSBedrockAgent, CloudAWSBedrockAgentConfig
from pipeline.enrichment.agents.cloud_anthropic_agent import CloudAnthropicAgent
from pipeline.enrichment.errors import ConfigurationError


# Legacy provider name mapping for backward compatibility
LEGACY_PROVIDER_MAP = {
    "openai": "cloud-openai",
    "claude": "cloud-anthropic",
    "bedrock": "cloud-aws-bedrock",
    "ollama": "local-ollama",
}


@dataclass
class AutoSelectionConfig:
    """Configuration for auto-selection behavior.
    
    Attributes:
        priority_order: List of providers to try in order (supports both new and legacy names)
        fallback_enabled: Whether to fall back to next provider if first fails
    """
    priority_order: List[str] = field(
        default_factory=lambda: ["cloud-openai", "cloud-anthropic", "cloud-aws-bedrock", "local-ollama"]
    )
    fallback_enabled: bool = True


class AgentFactory:
    """Factory for creating LLM agents with auto-selection support.
    
    This factory handles:
    - Agent instantiation based on provider name
    - Agent caching to prevent redundant instantiation
    - Auto-selection of available providers
    - Provider validation before use
    
    Example:
        >>> factory = AgentFactory(config)
        >>> agent = factory.create_agent("openai")
        >>> # Or use auto-selection
        >>> agent = factory.create_agent("auto")
    """
    
    def __init__(
        self,
        openai_config: Optional[CloudOpenAIAgentConfig] = None,
        ollama_config: Optional[LocalOllamaAgentConfig] = None,
        bedrock_config: Optional[CloudAWSBedrockAgentConfig] = None,
        claude_api_key: Optional[str] = None,
        auto_selection: Optional[AutoSelectionConfig] = None
    ):
        """Initialize agent factory.
        
        Args:
            openai_config: Configuration for CloudOpenAI agent
            ollama_config: Configuration for LocalOllama agent
            bedrock_config: Configuration for CloudAWSBedrock agent
            claude_api_key: API key for CloudAnthropic agent (reads from ANTHROPIC_API_KEY if not provided)
            auto_selection: Configuration for auto-selection behavior
        """
        self.openai_config = openai_config or self._default_openai_config()
        self.ollama_config = ollama_config or LocalOllamaAgentConfig()
        self.bedrock_config = bedrock_config or self._default_bedrock_config()
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.auto_selection = auto_selection or AutoSelectionConfig()
        
        # Cache for instantiated agents
        self._agent_cache: Dict[str, BaseLLMAgent] = {}
    
    def _default_openai_config(self) -> CloudOpenAIAgentConfig:
        """Create default OpenAI config from environment."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        return CloudOpenAIAgentConfig(api_key=api_key)
    
    def _default_bedrock_config(self) -> CloudAWSBedrockAgentConfig:
        """Create default Bedrock config from environment."""
        return CloudAWSBedrockAgentConfig(
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    
    def create_agent(self, provider: str) -> BaseLLMAgent:
        """Create agent for specified provider.
        
        Supports both new provider IDs (cloud-openai, cloud-anthropic, cloud-aws-bedrock, local-ollama)
        and legacy names (openai, claude, bedrock, ollama) for backward compatibility.
        
        Args:
            provider: Provider name (new or legacy format) or "auto" for auto-selection
            
        Returns:
            Instantiated LLM agent
            
        Raises:
            ConfigurationError: If provider is unknown or unavailable
        """
        if provider == "auto":
            return self._auto_select_agent()
        
        # Normalize provider name (handle legacy names)
        normalized_provider = LEGACY_PROVIDER_MAP.get(provider, provider)
        
        # Emit deprecation warning for legacy names
        if provider in LEGACY_PROVIDER_MAP:
            import warnings
            warnings.warn(
                f"Provider name '{provider}' is deprecated. "
                f"Use '{normalized_provider}' instead. "
                f"Legacy names will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Check cache first
        if normalized_provider in self._agent_cache:
            return self._agent_cache[normalized_provider]
        
        # Instantiate new agent
        agent = self._instantiate_agent(normalized_provider)
        
        # Cache for future use
        self._agent_cache[normalized_provider] = agent
        
        return agent
    
    def _auto_select_agent(self) -> BaseLLMAgent:
        """Auto-select first available provider.
        
        Tries providers in priority order and returns the first one that
        passes validation checks. Supports both new and legacy provider names.
        
        Returns:
            First available LLM agent
            
        Raises:
            ConfigurationError: If no providers are available
        """
        errors = []
        
        for provider in self.auto_selection.priority_order:
            # Normalize provider name (handle legacy names in config)
            normalized_provider = LEGACY_PROVIDER_MAP.get(provider, provider)
            
            try:
                agent = self._instantiate_agent(normalized_provider)
                
                # Validate that provider is available
                if agent.validate_requirements():
                    # Cache the selected agent
                    self._agent_cache[normalized_provider] = agent
                    return agent
                else:
                    errors.append(f"{normalized_provider}: validation failed")
                    
            except Exception as e:
                errors.append(f"{normalized_provider}: {str(e)}")
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
    
    def _instantiate_agent(self, provider: str) -> BaseLLMAgent:
        """Instantiate specific agent type.
        
        Args:
            provider: Provider name (new format: cloud-openai, cloud-anthropic, etc.)
            
        Returns:
            Instantiated agent
            
        Raises:
            ConfigurationError: If provider is unknown or not configured
        """
        if provider == "cloud-openai":
            if not self.openai_config.api_key:
                raise ConfigurationError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY environment variable or provide in config."
                )
            return CloudOpenAIAgent(self.openai_config)
        
        elif provider == "local-ollama":
            return LocalOllamaAgent(self.ollama_config)
        
        elif provider == "cloud-aws-bedrock":
            return CloudAWSBedrockAgent(self.bedrock_config)
        
        elif provider == "cloud-anthropic":
            if not self.claude_api_key:
                raise ConfigurationError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY environment variable or provide in config."
                )
            return CloudAnthropicAgent(api_key=self.claude_api_key)
        
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
                agent = self._instantiate_agent(provider)
                if agent.validate_requirements():
                    available.append(provider)
            except (ConfigurationError, NotImplementedError):
                continue
        
        return available
    
    def clear_cache(self):
        """Clear the agent cache.
        
        This forces re-instantiation of agents on next request.
        Useful for testing or when configuration changes.
        """
        self._agent_cache.clear()
