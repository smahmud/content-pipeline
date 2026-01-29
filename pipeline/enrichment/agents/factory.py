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
from pipeline.enrichment.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
from pipeline.enrichment.agents.ollama_agent import OllamaAgent, OllamaAgentConfig
from pipeline.enrichment.agents.bedrock_agent import BedrockAgent, BedrockAgentConfig
from pipeline.enrichment.agents.claude_agent import ClaudeAgent
from pipeline.enrichment.errors import ConfigurationError


@dataclass
class AutoSelectionConfig:
    """Configuration for auto-selection behavior.
    
    Attributes:
        priority_order: List of providers to try in order
        fallback_enabled: Whether to fall back to next provider if first fails
    """
    priority_order: List[str] = field(
        default_factory=lambda: ["openai", "claude", "bedrock", "ollama"]
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
        openai_config: Optional[OpenAIAgentConfig] = None,
        ollama_config: Optional[OllamaAgentConfig] = None,
        bedrock_config: Optional[BedrockAgentConfig] = None,
        claude_api_key: Optional[str] = None,
        auto_selection: Optional[AutoSelectionConfig] = None
    ):
        """Initialize agent factory.
        
        Args:
            openai_config: Configuration for OpenAI agent
            ollama_config: Configuration for Ollama agent
            bedrock_config: Configuration for Bedrock agent
            claude_api_key: API key for Claude agent (reads from ANTHROPIC_API_KEY if not provided)
            auto_selection: Configuration for auto-selection behavior
        """
        self.openai_config = openai_config or self._default_openai_config()
        self.ollama_config = ollama_config or OllamaAgentConfig()
        self.bedrock_config = bedrock_config or self._default_bedrock_config()
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.auto_selection = auto_selection or AutoSelectionConfig()
        
        # Cache for instantiated agents
        self._agent_cache: Dict[str, BaseLLMAgent] = {}
    
    def _default_openai_config(self) -> OpenAIAgentConfig:
        """Create default OpenAI config from environment."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        return OpenAIAgentConfig(api_key=api_key)
    
    def _default_bedrock_config(self) -> BedrockAgentConfig:
        """Create default Bedrock config from environment."""
        return BedrockAgentConfig(
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    
    def create_agent(self, provider: str) -> BaseLLMAgent:
        """Create agent for specified provider.
        
        Args:
            provider: Provider name ("openai", "ollama", "bedrock", "claude", "auto")
            
        Returns:
            Instantiated LLM agent
            
        Raises:
            ConfigurationError: If provider is unknown or unavailable
        """
        if provider == "auto":
            return self._auto_select_agent()
        
        # Check cache first
        if provider in self._agent_cache:
            return self._agent_cache[provider]
        
        # Instantiate new agent
        agent = self._instantiate_agent(provider)
        
        # Cache for future use
        self._agent_cache[provider] = agent
        
        return agent
    
    def _auto_select_agent(self) -> BaseLLMAgent:
        """Auto-select first available provider.
        
        Tries providers in priority order and returns the first one that
        passes validation checks.
        
        Returns:
            First available LLM agent
            
        Raises:
            ConfigurationError: If no providers are available
        """
        errors = []
        
        for provider in self.auto_selection.priority_order:
            try:
                agent = self._instantiate_agent(provider)
                
                # Validate that provider is available
                if agent.validate_requirements():
                    # Cache the selected agent
                    self._agent_cache[provider] = agent
                    return agent
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
            "  - OpenAI: Set OPENAI_API_KEY environment variable\n"
            "  - Claude: Set ANTHROPIC_API_KEY environment variable\n"
            "  - Bedrock: Configure AWS credentials\n"
            "  - Ollama: Start local service with 'ollama serve'"
        )
    
    def _instantiate_agent(self, provider: str) -> BaseLLMAgent:
        """Instantiate specific agent type.
        
        Args:
            provider: Provider name
            
        Returns:
            Instantiated agent
            
        Raises:
            ConfigurationError: If provider is unknown or not configured
        """
        if provider == "openai":
            if not self.openai_config.api_key:
                raise ConfigurationError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY environment variable or provide in config."
                )
            return OpenAIAgent(self.openai_config)
        
        elif provider == "ollama":
            return OllamaAgent(self.ollama_config)
        
        elif provider == "bedrock":
            return BedrockAgent(self.bedrock_config)
        
        elif provider == "claude":
            if not self.claude_api_key:
                raise ConfigurationError(
                    "Claude API key not configured. "
                    "Set ANTHROPIC_API_KEY environment variable or provide in config."
                )
            return ClaudeAgent(api_key=self.claude_api_key)
        
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider}. "
                f"Valid options: openai, ollama, bedrock, claude, auto"
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently available providers.
        
        Returns:
            List of provider names that pass validation
        """
        available = []
        
        for provider in ["openai", "ollama", "bedrock", "claude"]:
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
