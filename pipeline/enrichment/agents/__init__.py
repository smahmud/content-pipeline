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
from pipeline.enrichment.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
from pipeline.enrichment.agents.ollama_agent import OllamaAgent, OllamaAgentConfig
from pipeline.enrichment.agents.bedrock_agent import BedrockAgent, BedrockAgentConfig

__all__ = [
    "BaseLLMAgent",
    "LLMRequest",
    "LLMResponse",
    "AgentFactory",
    "AutoSelectionConfig",
    "OpenAIAgent",
    "OpenAIAgentConfig",
    "OllamaAgent",
    "OllamaAgentConfig",
    "BedrockAgent",
    "BedrockAgentConfig",
]
