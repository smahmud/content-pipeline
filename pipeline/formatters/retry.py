"""
Formatter Retry Logic

Provides retry utilities for the formatter module, reusing the
retry infrastructure from the enrichment module.

This module wraps the enrichment retry logic and adds formatter-specific
error handling for graceful degradation.
"""

import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any

# Reuse retry infrastructure from enrichment module
from pipeline.enrichment.retry import (
    retry_with_backoff as enrichment_retry_with_backoff,
    calculate_backoff_delay,
    RetryContext,
    TRANSIENT_ERRORS as ENRICHMENT_TRANSIENT_ERRORS,
    PERMANENT_ERRORS as ENRICHMENT_PERMANENT_ERRORS,
)

# Import error classes from the new LLM infrastructure layer
from pipeline.llm.errors import (
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
)

from pipeline.formatters.errors import EnhancementError


logger = logging.getLogger(__name__)


# Re-export transient and permanent errors for formatter use
TRANSIENT_ERRORS = ENRICHMENT_TRANSIENT_ERRORS
PERMANENT_ERRORS = ENRICHMENT_PERMANENT_ERRORS


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.
    
    Transient errors include:
    - Rate limit errors (429)
    - Timeout errors
    - Network connectivity errors
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is transient, False otherwise
    """
    return isinstance(error, TRANSIENT_ERRORS)


def is_permanent_error(error: Exception) -> bool:
    """Check if an error is permanent and should NOT be retried.
    
    Permanent errors include:
    - Authentication errors (401, 403)
    - Invalid request errors (400)
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is permanent, False otherwise
    """
    return isinstance(error, PERMANENT_ERRORS)


def retry_enhancement(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    log_retries: bool = True,
):
    """Decorator for retrying LLM enhancement with exponential backoff.
    
    This decorator implements the retry behavior specified in Property 12:
    - Exponential backoff with delays of 1s, 2s, 4s
    - Maximum 3 retries for transient errors
    - No retry for authentication/invalid request errors
    
    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 1.0)
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_enhancement(max_attempts=3, base_delay=1.0)
        ... def enhance_content(content: str) -> str:
        ...     # LLM call that might fail
        ...     pass
    """
    return enrichment_retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        transient_errors=TRANSIENT_ERRORS,
        permanent_errors=PERMANENT_ERRORS,
        log_retries=log_retries,
    )


def retry_with_fallback(
    fallback_value: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    log_retries: bool = True,
    on_fallback: Optional[Callable[[Exception], None]] = None,
):
    """Decorator for retrying with fallback on exhausted retries.
    
    This decorator implements graceful degradation:
    - Retries transient errors with exponential backoff
    - Returns fallback value if all retries are exhausted
    - Calls on_fallback callback when falling back
    
    Args:
        fallback_value: Value to return if all retries fail
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 1.0)
        log_retries: Whether to log retry attempts
        on_fallback: Optional callback when fallback is used
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_with_fallback(fallback_value="original content")
        ... def enhance_content(content: str) -> str:
        ...     # LLM call that might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except PERMANENT_ERRORS as e:
                    # Permanent error - don't retry, use fallback
                    if log_retries:
                        logger.error(
                            f"Permanent error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                    last_exception = e
                    break
                
                except TRANSIENT_ERRORS as e:
                    last_exception = e
                    
                    # Check if we have more attempts
                    if attempt < max_attempts - 1:
                        # Calculate delay
                        delay = calculate_backoff_delay(attempt, base_delay)
                        
                        if log_retries:
                            logger.warning(
                                f"Transient error in {func.__name__} "
                                f"(attempt {attempt + 1}/{max_attempts}): "
                                f"{type(e).__name__}: {e}. "
                                f"Retrying in {delay}s..."
                            )
                        
                        # Wait before retry
                        time.sleep(delay)
                    else:
                        # Last attempt failed
                        if log_retries:
                            logger.warning(
                                f"All {max_attempts} attempts failed for {func.__name__}: "
                                f"{type(e).__name__}: {e}. Using fallback."
                            )
                
                except Exception as e:
                    # Unknown error - don't retry, use fallback
                    if log_retries:
                        logger.error(
                            f"Unknown error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                    last_exception = e
                    break
            
            # All retries exhausted or permanent error - use fallback
            if on_fallback and last_exception:
                on_fallback(last_exception)
            
            return fallback_value
        
        return wrapper
    return decorator


class EnhancementRetryContext:
    """Context manager for enhancement retry logic with graceful degradation.
    
    This provides fine-grained control over retry logic for LLM enhancement,
    including tracking of attempts and support for graceful degradation.
    
    Example:
        >>> ctx = EnhancementRetryContext(max_attempts=3)
        >>> for attempt in ctx:
        ...     try:
        ...         result = enhance_with_llm(content)
        ...         break  # Success
        ...     except RateLimitError as e:
        ...         if not ctx.should_retry(e):
        ...             # Use fallback
        ...             result = content
        ...             ctx.mark_fallback_used(e)
        ...             break
        ...         ctx.wait()
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        log_retries: bool = True,
    ):
        """Initialize enhancement retry context.
        
        Args:
            max_attempts: Maximum number of attempts
            base_delay: Base delay for exponential backoff
            log_retries: Whether to log retry attempts
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.log_retries = log_retries
        self.current_attempt = 0
        self.last_error: Optional[Exception] = None
        self.fallback_used = False
        self.fallback_reason: Optional[str] = None
    
    def __iter__(self):
        """Iterate over retry attempts."""
        self.current_attempt = 0
        self.last_error = None
        self.fallback_used = False
        self.fallback_reason = None
        return self
    
    def __next__(self):
        """Get next retry attempt."""
        if self.current_attempt >= self.max_attempts:
            raise StopIteration
        
        attempt = self.current_attempt
        self.current_attempt += 1
        return attempt
    
    def should_retry(self, error: Exception) -> bool:
        """Check if error should trigger retry.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        self.last_error = error
        
        # Don't retry permanent errors
        if is_permanent_error(error):
            return False
        
        # Don't retry if we've exhausted attempts
        if self.current_attempt >= self.max_attempts:
            return False
        
        # Retry transient errors
        if is_transient_error(error):
            return True
        
        # Don't retry unknown errors
        return False
    
    def wait(self):
        """Wait with exponential backoff before next retry."""
        import time
        
        if self.current_attempt > 0:
            delay = calculate_backoff_delay(self.current_attempt - 1, self.base_delay)
            
            if self.log_retries:
                logger.info(f"Waiting {delay}s before retry...")
            
            time.sleep(delay)
    
    def mark_fallback_used(self, error: Exception):
        """Mark that fallback was used due to error.
        
        Args:
            error: The error that triggered fallback
        """
        self.fallback_used = True
        self.last_error = error
        self.fallback_reason = f"{type(error).__name__}: {error}"
        
        if self.log_retries:
            logger.warning(
                f"Using fallback after {self.current_attempt} attempts: {self.fallback_reason}"
            )
    
    def get_enhancement_error(self) -> Optional[EnhancementError]:
        """Get EnhancementError if fallback was used.
        
        Returns:
            EnhancementError if fallback was used, None otherwise
        """
        if not self.fallback_used or not self.last_error:
            return None
        
        return EnhancementError(
            message=f"LLM enhancement failed after {self.current_attempt} attempts",
            original_error=self.last_error,
            attempts=self.current_attempt,
            recoverable=True,
        )


def get_retry_delays(max_attempts: int = 3, base_delay: float = 1.0) -> list:
    """Get the sequence of retry delays for documentation/testing.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay for exponential backoff
        
    Returns:
        List of delays in seconds (one less than max_attempts since
        no delay after last attempt)
        
    Example:
        >>> get_retry_delays(3, 1.0)
        [1.0, 2.0]  # Delays before attempt 2 and 3
    """
    return [calculate_backoff_delay(i, base_delay) for i in range(max_attempts - 1)]
