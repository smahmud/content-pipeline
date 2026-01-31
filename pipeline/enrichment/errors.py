"""
Enrichment Error Hierarchy

Defines all custom exceptions used in the enrichment system.
This provides clear, specific error types for different failure scenarios.
"""


class EnrichmentError(Exception):
    """Base exception for all enrichment errors."""
    pass


class LLMProviderError(EnrichmentError):
    """Error communicating with LLM provider.
    
    Raised when an API call to an LLM provider fails due to network issues,
    service unavailability, or other provider-specific errors.
    """
    pass


class AuthenticationError(LLMProviderError):
    """Authentication with LLM provider failed.
    
    Raised when API keys are invalid, expired, or missing.
    """
    pass


class RateLimitError(LLMProviderError):
    """Rate limit exceeded for LLM provider.
    
    Raised when too many requests are made in a short time period.
    This is a transient error that should trigger retry logic.
    """
    pass


class PromptRenderError(EnrichmentError):
    """Error rendering prompt template.
    
    Raised when a Jinja2 template fails to render due to syntax errors
    or missing variables.
    """
    pass


class CostLimitExceededError(EnrichmentError):
    """Estimated cost exceeds user-specified limit.
    
    Raised when the estimated cost of an enrichment operation exceeds
    the --max-cost limit specified by the user.
    """
    pass


class ContextWindowExceededError(EnrichmentError):
    """Transcript exceeds LLM context window even after chunking.
    
    Raised when a transcript is too large to process even after
    attempting to split it into chunks.
    """
    pass


class SchemaValidationError(EnrichmentError):
    """LLM response does not conform to expected schema.
    
    Raised when an LLM response cannot be parsed or validated against
    the expected Pydantic schema.
    
    Attributes:
        enrichment_type: Type of enrichment that failed validation
        response_text: The raw LLM response text
        original_error: The underlying validation or parsing error
    """
    
    def __init__(
        self,
        message: str,
        enrichment_type: str = None,
        response_text: str = None,
        original_error: Exception = None
    ):
        super().__init__(message)
        self.enrichment_type = enrichment_type
        self.response_text = response_text
        self.original_error = original_error


class CacheError(EnrichmentError):
    """Error reading or writing cache.
    
    Raised when cache operations fail due to file system issues,
    permissions, or corrupted cache data.
    """
    pass


class PromptTemplateError(EnrichmentError):
    """Error loading or validating prompt template.
    
    Raised when a YAML prompt template is malformed, missing required
    fields, or cannot be loaded.
    """
    pass


class ConfigurationError(EnrichmentError):
    """Error in enrichment configuration.
    
    Raised when configuration is invalid, missing required fields,
    or contains conflicting settings.
    """
    pass



class InvalidRequestError(LLMProviderError):
    """Invalid request to LLM provider.
    
    Raised when a request is malformed or contains invalid parameters.
    This is a permanent error that should NOT trigger retry logic.
    """
    pass


class TimeoutError(LLMProviderError):
    """Request to LLM provider timed out.
    
    Raised when an API call takes longer than the configured timeout.
    This is a transient error that should trigger retry logic.
    """
    pass


class NetworkError(LLMProviderError):
    """Network error communicating with LLM provider.
    
    Raised when network connectivity issues prevent API calls.
    This is a transient error that should trigger retry logic.
    """
    pass


class ChunkingError(EnrichmentError):
    """Error during transcript chunking.
    
    Raised when automatic chunking fails or produces invalid chunks.
    """
    pass


class BatchProcessingError(EnrichmentError):
    """Error during batch processing.
    
    Raised when batch processing encounters an error that affects
    the entire batch operation.
    """
    pass


class OutputFileError(EnrichmentError):
    """Error writing output file.
    
    Raised when output file cannot be written due to permissions,
    disk space, or other file system issues.
    """
    pass
