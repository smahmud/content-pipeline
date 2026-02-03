"""
Cloud OpenAI LLM Provider

Cloud-based provider for OpenAI's GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo).
Handles authentication, API calls, token counting with tiktoken, and cost estimation.

File naming follows pattern: cloud_{service}.py
Provider ID: cloud-openai
"""

from typing import Dict, Any

from pipeline.llm.config import OpenAIConfig
from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse
from pipeline.llm.errors import (
    ProviderError,
    ProviderNotAvailableError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    TimeoutError,
    NetworkError
)
from pipeline.enrichment.retry import retry_with_backoff


class CloudOpenAIProvider(BaseLLMProvider):
    """Cloud OpenAI LLM provider.
    
    This provider integrates with OpenAI's cloud API to provide LLM capabilities.
    It supports GPT-3.5-turbo, GPT-4, and GPT-4-turbo models with accurate
    token counting using tiktoken and cost estimation based on current pricing.
    
    Configuration is loaded from:
    1. Environment variables (OPENAI_API_KEY, OPENAI_MODEL, etc.)
    2. Config file (config.yaml llm.openai section)
    3. Defaults (gpt-4-turbo, 4096 max_tokens, etc.)
    
    Deployment: Cloud (requires API key and internet connection)
    Provider: OpenAI
    Access Method: Direct API
    
    Example:
        >>> from pipeline.llm.config import OpenAIConfig
        >>> from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
        >>> 
        >>> # Load configuration from environment/config/defaults
        >>> config = OpenAIConfig.load_from_config()
        >>> 
        >>> # Create provider
        >>> provider = CloudOpenAIProvider(config)
        >>> 
        >>> # Generate completion
        >>> request = LLMRequest(
        >>>     prompt="Explain quantum computing",
        >>>     max_tokens=500,
        >>>     temperature=0.7
        >>> )
        >>> response = provider.generate(request)
        >>> print(response.content)
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
    
    def __init__(self, config: OpenAIConfig):
        """Initialize Cloud OpenAI provider.
        
        Args:
            config: OpenAI configuration loaded from environment/config/defaults
        """
        self.config = config
        self._client = None
        self._tiktoken_encoding = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client.
        
        Returns:
            Configured OpenAI client instance
            
        Raises:
            ImportError: If OpenAI SDK is not installed
        """
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
        """Get tiktoken encoding for the model.
        
        Args:
            model: Model identifier
            
        Returns:
            Tiktoken encoding instance
            
        Raises:
            ImportError: If tiktoken is not installed
        """
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
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from OpenAI.
        
        This method includes automatic retry logic with exponential backoff
        for transient errors (rate limits, timeouts, network issues).
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response with content and metadata
            
        Raises:
            RateLimitError: If rate limit is exceeded (transient, will retry)
            AuthenticationError: If authentication fails (permanent, no retry)
            InvalidRequestError: If request is invalid (permanent, no retry)
            TimeoutError: If request times out (transient, will retry)
            NetworkError: If network error occurs (transient, will retry)
            ProviderError: For other provider errors
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
            error_msg = str(e).lower()
            
            # Classify error type for proper retry behavior
            if "rate" in error_msg or "429" in error_msg:
                raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
            elif "auth" in error_msg or "401" in error_msg or "403" in error_msg:
                raise AuthenticationError(f"OpenAI authentication failed: {str(e)}")
            elif "invalid" in error_msg or "400" in error_msg:
                raise InvalidRequestError(f"Invalid OpenAI request: {str(e)}")
            elif "timeout" in error_msg:
                raise TimeoutError(f"OpenAI request timed out: {str(e)}")
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to OpenAI: {str(e)}")
            else:
                raise ProviderError(f"OpenAI API call failed: {str(e)}")
    
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
            Dictionary with provider capabilities including:
            - provider: Provider identifier
            - supported_models: List of supported model names
            - max_tokens: Maximum context window size
            - supports_streaming: Whether streaming is supported
            - supports_functions: Whether function calling is supported
            - supports_vision: Whether vision/image inputs are supported
        """
        return {
            "provider": "cloud-openai",
            "supported_models": list(self.PRICING.keys()),
            "max_tokens": max(self.CONTEXT_WINDOWS.values()),
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": False,  # Not implemented yet
        }
    
    def validate_requirements(self) -> bool:
        """Check if OpenAI is available.
        
        Performs a health check by attempting to list available models.
        
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
