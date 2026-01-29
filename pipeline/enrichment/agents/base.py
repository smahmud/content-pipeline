"""
Base LLM Agent Protocol

Defines the abstract base class and standardized request/response formats
for all LLM provider adapters. This ensures a consistent interface across
OpenAI, AWS Bedrock, Claude, and Ollama providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LLMRequest:
    """Standardized request format for all LLM agents.
    
    This dataclass encapsulates all parameters needed to make an LLM API call,
    providing a consistent interface across different providers.
    
    Attributes:
        prompt: The complete prompt text to send to the LLM
        max_tokens: Maximum number of tokens to generate in the response
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        model: Optional specific model to use (overrides agent default)
        metadata: Additional provider-specific parameters
    """
    prompt: str
    max_tokens: int
    temperature: float
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")


@dataclass
class LLMResponse:
    """Standardized response format from all LLM agents.
    
    This dataclass encapsulates the LLM response along with metadata about
    the API call, including cost and token usage information.
    
    Attributes:
        content: The generated text content from the LLM
        model_used: The actual model that processed the request
        tokens_used: Total number of tokens consumed (input + output)
        cost_usd: Estimated cost in USD for this API call
        metadata: Additional provider-specific response data
    """
    content: str
    model_used: str
    tokens_used: int
    cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMAgent(ABC):
    """Abstract base class for LLM provider agents.
    
    This class defines the contract that all LLM provider adapters must implement.
    It ensures consistent behavior across OpenAI, AWS Bedrock, Claude, and Ollama
    providers while allowing for provider-specific implementations.
    
    The agent is responsible for:
    - Making API calls to the LLM provider
    - Estimating costs before making calls
    - Reporting provider capabilities
    - Validating that the provider is available
    - Determining context window sizes
    """
    
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM.
        
        This method sends a request to the LLM provider and returns the response.
        It should handle all provider-specific API details, authentication,
        and response formatting.
        
        Args:
            request: Standardized LLM request with prompt and parameters
            
        Returns:
            Standardized LLM response with content and metadata
            
        Raises:
            LLMProviderError: If the API call fails
            AuthenticationError: If credentials are invalid
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost in USD for the request.
        
        This method calculates the expected cost before making an API call,
        allowing users to make informed decisions about whether to proceed.
        
        Args:
            request: The LLM request to estimate cost for
            
        Returns:
            Estimated cost in USD (0.0 for local models like Ollama)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities.
        
        This method returns information about what the provider supports,
        including available models, maximum context windows, and special features.
        
        Returns:
            Dictionary containing:
                - models: List of available model identifiers
                - max_tokens: Maximum context window size
                - supports_streaming: Whether streaming is supported
                - supports_functions: Whether function calling is supported
        """
        pass
    
    @abstractmethod
    def validate_requirements(self) -> bool:
        """Check if provider is available.
        
        This method verifies that all requirements for using the provider
        are met, including API keys, network connectivity, and service health.
        
        Returns:
            True if provider is ready to use, False otherwise
        """
        pass
    
    @abstractmethod
    def get_context_window(self, model: str) -> int:
        """Return maximum context window size for the model.
        
        This method returns the maximum number of tokens that can be processed
        in a single request for the specified model. This is used to determine
        when transcript chunking is necessary.
        
        Args:
            model: The model identifier to check
            
        Returns:
            Maximum context window size in tokens
            
        Raises:
            ValueError: If the model is not supported by this provider
        """
        pass
