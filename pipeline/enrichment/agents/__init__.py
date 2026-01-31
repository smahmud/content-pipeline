"""
LLM Agent Adapters

This module provides adapters for different LLM providers, following a unified
interface defined by BaseLLMAgent. Each adapter handles provider-specific
authentication, API calls, cost estimation, and response formatting.
"""

from pipeline.enrichment.agents.base import (
    BaseLLMAgent,
    LLMRequest,
    LLMResponse,
)
from pipeline.enrichment.agents.factory import AgentFactory, AutoSelectionConfig
from pipeline.enrichment.agents.cloud_openai_agent import CloudOpenAIAgent, CloudOpenAIAgentConfig
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent, LocalOllamaAgentConfig
from pipeline.enrichment.agents.cloud_aws_bedrock_agent import CloudAWSBedrockAgent, CloudAWSBedrockAgentConfig
from pipeline.enrichment.agents.cloud_anthropic_agent import CloudAnthropicAgent

# Legacy imports for backward compatibility (deprecated)
# These will be removed in v1.0.0
OpenAIAgent = CloudOpenAIAgent
OpenAIAgentConfig = CloudOpenAIAgentConfig
OllamaAgent = LocalOllamaAgent
OllamaAgentConfig = LocalOllamaAgentConfig
BedrockAgent = CloudAWSBedrockAgent
BedrockAgentConfig = CloudAWSBedrockAgentConfig
ClaudeAgent = CloudAnthropicAgent

__all__ = [
    # Base classes
    "BaseLLMAgent",
    "LLMRequest",
    "LLMResponse",
    # Factory
    "AgentFactory",
    "AutoSelectionConfig",
    # New agent classes (preferred)
    "CloudOpenAIAgent",
    "CloudOpenAIAgentConfig",
    "LocalOllamaAgent",
    "LocalOllamaAgentConfig",
    "CloudAWSBedrockAgent",
    "CloudAWSBedrockAgentConfig",
    "CloudAnthropicAgent",
    # Legacy names (deprecated, for backward compatibility)
    "OpenAIAgent",
    "OpenAIAgentConfig",
    "OllamaAgent",
    "OllamaAgentConfig",
    "BedrockAgent",
    "BedrockAgentConfig",
    "ClaudeAgent",
]
