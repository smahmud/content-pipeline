"""
Anthropic Claude Agent

LLM agent implementation for Anthropic's Claude models via the Anthropic API.
Supports Claude 2, Claude 3 (Opus, Sonnet, Haiku) with proper prompt formatting
and cost estimation.
"""

from typing import Dict, Any, Optional
import os

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse
from pipeline.enrichment.retry import retry_with_backoff
from pipeline.enrichment.errors import (
    LLMProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError
)


class ClaudeAgent(BaseLLMAgent):
    """LLM agent for Anthropic Claude models.
    
    This agent interfaces with Anthropic's Claude API, supporting:
    - Claude 2 (claude-2.1, claude-2.0)
    - Claude 3 Opus (claude-3-opus-20240229)
    - Claude 3 Sonnet (claude-3-sonnet-20240229)
    - Claude 3 Haiku (claude-3-haiku-20240307)
    
    Features:
    - Proper prompt formatting for Claude's message format
    - Cost estimation based on Anthropic pricing
    - Context window management (up to 200K tokens)
    - Error handling with retry logic
    
    Example:
        >>> agent = ClaudeAgent(api_key="sk-ant-...")
        >>> request = LLMRequest(prompt="Summarize this text", max_tokens=500)
        >>> response = agent.generate(request)
    """
    
    # Pricing per 1M tokens (as of January 2026)
    PRICING = {
        # Claude 2
        "claude-2.1": {"input": 8.00, "output": 24.00},
        "claude-2.0": {"input": 8.00, "output": 24.00},
        
        # Claude 3 Opus
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        
        # Claude 3 Sonnet
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        
        # Claude 3 Haiku
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # Context windows (in tokens)
    CONTEXT_WINDOWS = {
        "claude-2.1": 200_000,
        "claude-2.0": 100_000,
        "claude-3-opus-20240229": 200_000,
        "claude-3-sonnet-20240229": 200_000,
        "claude-3-haiku-20240307": 200_000,
    }
    
    # Default model
    DEFAULT_MODEL = "claude-3-sonnet-20240229"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """Initialize Claude agent.
        
        Args:
            api_key: Anthropic API key (reads from ANTHROPIC_API_KEY env var if not provided)
            default_model: Default model to use (default: claude-3-sonnet-20240229)
            
        Raises:
            AuthenticationError: If API key is not provided or invalid
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self.default_model = default_model or self.DEFAULT_MODEL
        
        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise LLMProviderError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Claude.
        
        Args:
            request: LLM request with prompt and parameters
            
        Returns:
            LLM response with generated content
            
        Raises:
            LLMProviderError: If API call fails
            RateLimitError: If rate limit is exceeded
            InvalidRequestError: If request is invalid
        """
        model = request.model or self.default_model
        
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
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages
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
                cost_usd=cost
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
                raise LLMProviderError(f"Claude API error: {error_msg}")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for a request.
        
        Args:
            request: LLM request to estimate cost for
            
        Returns:
            Estimated cost in USD
        """
        model = request.model or self.default_model
        
        # Estimate input tokens (rough approximation: 1 token ≈ 0.75 words)
        input_tokens = int(len(request.prompt.split()) * 1.3)
        
        # Use max_tokens as output estimate
        output_tokens = request.max_tokens
        
        return self._calculate_cost(model, input_tokens, output_tokens)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities.
        
        Returns:
            Dict with provider info, supported models, and features
        """
        return {
            "provider": "claude",
            "default_model": self.default_model,
            "supported_models": list(self.PRICING.keys()),
            "max_context_window": max(self.CONTEXT_WINDOWS.values()),
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_vision": True  # Claude 3 models support vision
        }
    
    def validate_requirements(self, request: LLMRequest) -> bool:
        """Validate that request meets agent requirements.
        
        Args:
            request: LLM request to validate
            
        Returns:
            True if valid
            
        Raises:
            InvalidRequestError: If request is invalid
        """
        model = request.model or self.default_model
        
        # Check if model is supported
        if model not in self.PRICING:
            raise InvalidRequestError(
                f"Unsupported Claude model: {model}"
            )
        
        # Check context window
        context_window = self.CONTEXT_WINDOWS.get(model, 200_000)
        estimated_tokens = int(len(request.prompt.split()) * 1.3)
        
        if estimated_tokens + request.max_tokens > context_window:
            raise InvalidRequestError(
                f"Request exceeds context window for {model}. "
                f"Estimated tokens: {estimated_tokens + request.max_tokens}, "
                f"Context window: {context_window}"
            )
        
        return True
    
    def get_context_window(self, model: Optional[str] = None) -> int:
        """Get context window size for model.
        
        Args:
            model: Model identifier (uses default if not specified)
            
        Returns:
            Context window size in tokens
        """
        model = model or self.default_model
        return self.CONTEXT_WINDOWS.get(model, 200_000)
    
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
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0
        
        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
