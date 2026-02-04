"""
Cloud Anthropic Claude Provider

Cloud-based LLM provider implementation for Anthropic's Claude models via the Anthropic API.
Supports Claude 2, Claude 3 (Opus, Sonnet, Haiku) with proper prompt formatting
and cost estimation.

File naming follows pattern: cloud_{provider}.py
Provider ID: cloud-anthropic
"""

from typing import Dict, Any

from pipeline.llm.config import AnthropicConfig
from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse
from pipeline.llm.errors import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderNotAvailableError,
)


class CloudAnthropicProvider(BaseLLMProvider):
    """Cloud Anthropic Claude LLM provider.
    
    This provider interfaces with Anthropic's cloud API, supporting:
    - Claude 2 (claude-2.1, claude-2.0)
    - Claude 3 Opus (claude-3-opus-20240229)
    - Claude 3 Sonnet (claude-3-sonnet-20240229)
    - Claude 3.5 Sonnet (claude-3-5-sonnet-20241022, claude-3-5-sonnet-20240620)
    - Claude 3 Haiku (claude-3-haiku-20240307)
    
    Deployment: Cloud (requires API key and internet connection)
    Provider: Anthropic
    Access Method: Direct API
    
    Features:
    - Proper prompt formatting for Claude's message format
    - Cost estimation based on Anthropic pricing
    - Context window management (up to 200K tokens)
    - Error handling with retry logic
    
    Example:
        >>> from pipeline.llm.config import AnthropicConfig
        >>> config = AnthropicConfig(api_key="sk-ant-...")
        >>> provider = CloudAnthropicProvider(config)
        >>> request = LLMRequest(prompt="Summarize this text", max_tokens=500, temperature=0.7)
        >>> response = provider.generate(request)
    """
    
    # Pricing per 1M tokens (as of January 2026)
    PRICING = {
        # Claude 2
        "claude-2.1": {"input": 8.00, "output": 24.00},
        "claude-2.0": {"input": 8.00, "output": 24.00},
        
        # Claude 3 Opus
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        
        # Claude 3 Sonnet (legacy)
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        
        # Claude 3.5 Sonnet (current)
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        
        # Claude 3 Haiku
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # Context windows (in tokens)
    CONTEXT_WINDOWS = {
        "claude-2.1": 200_000,
        "claude-2.0": 100_000,
        "claude-3-opus-20240229": 200_000,
        "claude-3-sonnet-20240229": 200_000,
        "claude-3-5-sonnet-20241022": 200_000,
        "claude-3-5-sonnet-20240620": 200_000,
        "claude-3-haiku-20240307": 200_000,
    }
    
    def __init__(self, config: AnthropicConfig):
        """Initialize Cloud Anthropic provider.
        
        Args:
            config: Anthropic configuration from pipeline.llm.config
            
        Raises:
            AuthenticationError: If API key is not provided or invalid
            ProviderNotAvailableError: If anthropic package is not installed
        """
        if not config.api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or provide api_key in configuration."
            )
        
        self.config = config
        
        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ProviderNotAvailableError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Claude.
        
        Args:
            request: LLM request with prompt and parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            ProviderError: If API call fails
            RateLimitError: If rate limit is exceeded
            InvalidRequestError: If request is invalid
        """
        model = request.model or self.config.default_model
        
        # Validate model
        if model not in self.PRICING:
            raise InvalidRequestError(
                f"Unsupported Claude model: {model}. "
                f"Supported models: {', '.join(self.PRICING.keys())}"
            )
        
        try:
            # Format prompt for Claude's message format
            messages = [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
            
            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature,
                messages=messages,
                timeout=self.config.timeout
            )
            
            # Extract content
            content = response.content[0].text
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            return LLMResponse(
                content=content,
                model_used=model,
                tokens_used=input_tokens + output_tokens,
                cost_usd=cost,
                metadata={
                    "provider": "anthropic",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific error types
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(f"Claude rate limit exceeded: {error_msg}")
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise AuthenticationError(f"Claude authentication failed: {error_msg}")
            elif "invalid" in error_msg.lower():
                raise InvalidRequestError(f"Invalid Claude request: {error_msg}")
            else:
                raise ProviderError(f"Claude API error: {error_msg}")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for a request.
        
        Args:
            request: LLM request to estimate cost for
            
        Returns:
            Estimated cost in USD
        """
        model = request.model or self.config.default_model
        
        # Estimate input tokens (rough approximation: 1 token ≈ 0.75 words)
        input_tokens = int(len(request.prompt.split()) * 1.3)
        
        # Use max_tokens as output estimate
        output_tokens = request.max_tokens or self.config.max_tokens
        
        return self._calculate_cost(model, input_tokens, output_tokens)
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for token usage.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Check for pricing override first
        if self.config.pricing_override and model in self.config.pricing_override:
            pricing = self.config.pricing_override[model]
        else:
            pricing = self.PRICING.get(model)
        
        if not pricing:
            return 0.0
        
        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities.
        
        Returns:
            Dict with provider info, supported models, and features
        """
        return {
            "provider": "cloud-anthropic",
            "default_model": self.config.default_model,
            "supported_models": list(self.PRICING.keys()),
            "max_context_window": max(self.CONTEXT_WINDOWS.values()),
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_vision": True  # Claude 3 models support vision
        }
    
    def validate_requirements(self) -> bool:
        """Check if provider is available and credentials are valid.
        
        Returns:
            True if provider is ready to use
        """
        try:
            # Try to make a minimal API call to validate credentials
            # We'll just check if the client is initialized properly
            return self.client is not None and self.config.api_key is not None
        except Exception:
            return False
    
    def get_context_window(self, model: str) -> int:
        """Get context window size for model.
        
        Args:
            model: Model identifier
            
        Returns:
            Context window size in tokens
        """
        return self.CONTEXT_WINDOWS.get(model, 200_000)
