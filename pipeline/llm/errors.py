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
