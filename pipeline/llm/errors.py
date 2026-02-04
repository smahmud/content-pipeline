"""
LLM Infrastructure Error Classes

This module defines the exception hierarchy for LLM infrastructure errors.
All LLM-related errors inherit from LLMError base class, enabling
consistent error handling across all LLM providers.

Error Hierarchy:
    LLMError (base)
    ├── ConfigurationError (invalid/missing configuration)
    ├── ProviderError (provider operation failures)
    └── ProviderNotAvailableError (provider not accessible)

Usage:
    >>> from pipeline.llm.errors import ConfigurationError
    >>> 
    >>> if not api_key:
    >>>     raise ConfigurationError(
    >>>         "OpenAI API key not configured. "
    >>>         "Set OPENAI_API_KEY environment variable."
    >>>     )
"""


class LLMError(Exception):
    """Base exception for all LLM infrastructure errors.
    
    All LLM-related exceptions inherit from this class, enabling
    catch-all error handling when needed:
    
    Example:
        >>> try:
        >>>     provider.generate(request)
        >>> except LLMError as e:
        >>>     logger.error(f"LLM operation failed: {e}")
    """
    pass


class ConfigurationError(LLMError):
    """Raised when LLM configuration is invalid or missing.
    
    This error indicates that required configuration values are missing,
    invalid, or improperly formatted. The error message should provide
    clear guidance on how to fix the configuration issue.
    
    Common scenarios:
    - Missing API keys
    - Invalid base URLs
    - Out-of-range parameter values (temperature, max_tokens)
    - Missing required credentials
    
    Example:
        >>> raise ConfigurationError(
        >>>     "OpenAI API key not configured. "
        >>>     "Set OPENAI_API_KEY environment variable or "
        >>>     "add 'api_key' to config.yaml llm.openai section."
        >>> )
    """
    pass


class ProviderError(LLMError):
    """Raised when a provider operation fails.
    
    This error indicates that an LLM provider operation (generate,
    estimate_cost, etc.) failed during execution. The error message
    should include context about what operation failed and why.
    
    Common scenarios:
    - API request failures
    - Network connectivity issues
    - Rate limiting
    - Invalid request parameters
    - Model not found
    
    Example:
        >>> raise ProviderError(
        >>>     f"OpenAI API request failed: {response.status_code} "
        >>>     f"{response.text}"
        >>> )
    """
    pass


class ProviderNotAvailableError(LLMError):
    """Raised when a requested provider is not available.
    
    This error indicates that a provider cannot be used because it
    fails validation checks. This could be due to missing dependencies,
    unavailable services, or failed health checks.
    
    Common scenarios:
    - Service not running (e.g., Ollama not started)
    - Missing dependencies (e.g., package not installed)
    - Network unreachable
    - Invalid credentials
    - Service health check failed
    
    Example:
        >>> raise ProviderNotAvailableError(
        >>>     "Cannot connect to Ollama at http://localhost:11434. "
        >>>     "Is Ollama running? Start with: ollama serve"
        >>> )
    """
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded.
    
    This error indicates that too many requests have been made to the
    provider in a short time period. This is a transient error that
    should trigger retry logic with exponential backoff.
    
    Common scenarios:
    - API rate limits exceeded
    - Quota limits reached
    - Concurrent request limits exceeded
    
    Example:
        >>> raise RateLimitError(
        >>>     "OpenAI rate limit exceeded: 429 Too Many Requests. "
        >>>     "Retry after 60 seconds."
        >>> )
    """
    pass


class AuthenticationError(ProviderError):
    """Raised when provider authentication fails.
    
    This error indicates that authentication credentials are invalid,
    expired, or missing. This is a permanent error that should NOT
    trigger retry logic.
    
    Common scenarios:
    - Invalid API key
    - Expired credentials
    - Missing authentication headers
    - Insufficient permissions
    
    Example:
        >>> raise AuthenticationError(
        >>>     "OpenAI authentication failed: Invalid API key. "
        >>>     "Check OPENAI_API_KEY environment variable."
        >>> )
    """
    pass


class InvalidRequestError(ProviderError):
    """Raised when provider request is invalid.
    
    This error indicates that the request parameters are malformed,
    invalid, or not supported by the provider. This is a permanent
    error that should NOT trigger retry logic.
    
    Common scenarios:
    - Invalid model name
    - Out-of-range parameters (temperature, max_tokens)
    - Malformed request body
    - Unsupported features
    
    Example:
        >>> raise InvalidRequestError(
        >>>     "Invalid OpenAI request: Model 'gpt-5' not found. "
        >>>     "Available models: gpt-3.5-turbo, gpt-4, gpt-4-turbo"
        >>> )
    """
    pass


class TimeoutError(ProviderError):
    """Raised when provider request times out.
    
    This error indicates that a request took longer than the configured
    timeout period. This is a transient error that should trigger retry
    logic with exponential backoff.
    
    Common scenarios:
    - Network latency
    - Provider service slowness
    - Large request processing
    
    Example:
        >>> raise TimeoutError(
        >>>     "OpenAI request timed out after 60 seconds. "
        >>>     "Try increasing timeout or reducing request size."
        >>> )
    """
    pass


class NetworkError(ProviderError):
    """Raised when network connectivity issues occur.
    
    This error indicates that network connectivity problems prevented
    the request from completing. This is a transient error that should
    trigger retry logic with exponential backoff.
    
    Common scenarios:
    - DNS resolution failures
    - Connection refused
    - Network unreachable
    - SSL/TLS errors
    
    Example:
        >>> raise NetworkError(
        >>>     "Network error connecting to OpenAI: "
        >>>     "Connection refused. Check internet connectivity."
        >>> )
    """
    pass
