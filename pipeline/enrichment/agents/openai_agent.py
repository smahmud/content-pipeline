"""
OpenAI LLM Agent

Adapter for OpenAI's GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo).
Handles authentication, API calls, token counting with tiktoken, and cost estimation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse


@dataclass
class OpenAIAgentConfig:
    """Configuration for OpenAI agent.
    
    Attributes:
        api_key: OpenAI API key (can be loaded from environment)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        timeout: Request timeout in seconds
    """
    api_key: str
    default_model: str = "gpt-4-turbo"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 60


class OpenAIAgent(BaseLLMAgent):
    """OpenAI LLM provider adapter.
    
    This agent integrates with OpenAI's API to provide LLM capabilities.
    It supports GPT-3.5-turbo, GPT-4, and GPT-4-turbo models with accurate
    token counting using tiktoken and cost estimation based on current pricing.
    """
    
    # Pricing database (USD per 1K tokens) - Updated as of January 2026
    PRICING = {
        "gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
        "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
        "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
        "gpt-4-turbo-preview": {"input_per_1k": 0.01, "output_per_1k": 0.03},
    }
    
    # Context window sizes (in tokens)
    CONTEXT_WINDOWS = {
        "gpt-3.5-turbo": 16385,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-turbo-preview": 128000,
    }
    
    def __init__(self, config: OpenAIAgentConfig):
        """Initialize OpenAI agent.
        
        Args:
            config: Configuration for the OpenAI agent
        """
        self.config = config
        self._client = None
        self._tiktoken_encoding = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. Install with: pip install openai"
                )
        return self._client
    
    def _get_tiktoken_encoding(self, model: str):
        """Get tiktoken encoding for the model."""
        if self._tiktoken_encoding is None:
            try:
                import tiktoken
                # Use cl100k_base encoding for GPT-4 and GPT-3.5-turbo
                self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                raise ImportError(
                    "tiktoken not installed. Install with: pip install tiktoken"
                )
        return self._tiktoken_encoding
    
    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model identifier (for encoding selection)
            
        Returns:
            Number of tokens in the text
        """
        encoding = self._get_tiktoken_encoding(model)
        return len(encoding.encode(text))
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from OpenAI.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response with content and metadata
            
        Raises:
            LLMProviderError: If the API call fails
        """
        model = request.model or self.config.default_model
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": request.prompt}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                **request.metadata
            )
            
            # Extract response data
            content = response.choices[0].message.content
            
            # Calculate token usage
            input_tokens = self._count_tokens(request.prompt, model)
            output_tokens = self._count_tokens(content, model)
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            pricing = self.PRICING.get(model, self.PRICING["gpt-4-turbo"])
            input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
            output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
            total_cost = input_cost + output_cost
            
            return LLMResponse(
                content=content,
                model_used=model,
                tokens_used=total_tokens,
                cost_usd=total_cost,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                }
            )
            
        except Exception as e:
            # Import here to avoid circular dependency
            from pipeline.enrichment.errors import LLMProviderError
            raise LLMProviderError(f"OpenAI API call failed: {str(e)}")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for the request.
        
        Args:
            request: The LLM request to estimate cost for
            
        Returns:
            Estimated cost in USD
        """
        model = request.model or self.config.default_model
        
        # Count input tokens
        input_tokens = self._count_tokens(request.prompt, model)
        
        # Estimate output tokens (use max_tokens as upper bound)
        output_tokens = request.max_tokens
        
        # Get pricing
        pricing = self.PRICING.get(model, self.PRICING["gpt-4-turbo"])
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        
        return input_cost + output_cost
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return OpenAI provider capabilities.
        
        Returns:
            Dictionary with provider capabilities
        """
        return {
            "provider": "openai",
            "models": list(self.PRICING.keys()),
            "max_tokens": max(self.CONTEXT_WINDOWS.values()),
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": False,  # Not implemented yet
        }
    
    def validate_requirements(self) -> bool:
        """Check if OpenAI is available.
        
        Returns:
            True if OpenAI API is accessible, False otherwise
        """
        if not self.config.api_key:
            return False
        
        try:
            # Try to list models as a health check
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def get_context_window(self, model: str) -> int:
        """Return maximum context window size for the model.
        
        Args:
            model: The model identifier
            
        Returns:
            Maximum context window size in tokens
            
        Raises:
            ValueError: If the model is not supported
        """
        if model not in self.CONTEXT_WINDOWS:
            raise ValueError(
                f"Model '{model}' not supported. "
                f"Supported models: {list(self.CONTEXT_WINDOWS.keys())}"
            )
        return self.CONTEXT_WINDOWS[model]
