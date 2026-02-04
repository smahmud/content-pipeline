"""
Retry Logic with Exponential Backoff

Provides retry decorators and utilities for handling transient errors
in LLM API calls. Implements exponential backoff with configurable
retry attempts and delays.
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional

from pipeline.llm.errors import (
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError
)


logger = logging.getLogger(__name__)


# Transient errors that should trigger retry
TRANSIENT_ERRORS = (
    RateLimitError,
    TimeoutError,
    NetworkError,
)

# Permanent errors that should NOT trigger retry
PERMANENT_ERRORS = (
    AuthenticationError,
    InvalidRequestError,
)


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is transient, False otherwise
    """
    return isinstance(error, TRANSIENT_ERRORS)


def is_permanent_error(error: Exception) -> bool:
    """Check if an error is permanent and should NOT be retried.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is permanent, False otherwise
    """
    return isinstance(error, PERMANENT_ERRORS)


def calculate_backoff_delay(attempt: int, base_delay: float = 1.0) -> float:
    """Calculate exponential backoff delay.
    
    Uses exponential backoff: delay = base_delay * (2 ** attempt)
    - Attempt 0: 1s
    - Attempt 1: 2s
    - Attempt 2: 4s
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        
    Returns:
        Delay in seconds
    """
    return base_delay * (2 ** attempt)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    transient_errors: Tuple[Type[Exception], ...] = TRANSIENT_ERRORS,
    permanent_errors: Tuple[Type[Exception], ...] = PERMANENT_ERRORS,
    log_retries: bool = True
):
    """Decorator for retrying functions with exponential backoff.
    
    This decorator will retry a function if it raises a transient error,
    using exponential backoff between attempts. Permanent errors are
    raised immediately without retry.
    
    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 1.0)
        transient_errors: Tuple of exception types to retry
        permanent_errors: Tuple of exception types to NOT retry
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_with_backoff(max_attempts=3, base_delay=1.0)
        ... def call_api():
        ...     # API call that might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except permanent_errors as e:
                    # Permanent error - don't retry
                    if log_retries:
                        logger.error(
                            f"Permanent error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                    raise
                
                except transient_errors as e:
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
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}: "
                                f"{type(e).__name__}: {e}"
                            )
                
                except Exception as e:
                    # Unknown error - treat as permanent
                    if log_retries:
                        logger.error(
                            f"Unknown error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                    raise
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry logic with exponential backoff.
    
    This provides a more flexible alternative to the decorator for cases
    where you need fine-grained control over retry logic.
    
    Example:
        >>> retry_ctx = RetryContext(max_attempts=3, base_delay=1.0)
        >>> for attempt in retry_ctx:
        ...     try:
        ...         result = call_api()
        ...         break  # Success
        ...     except RateLimitError as e:
        ...         if not retry_ctx.should_retry(e):
        ...             raise
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        log_retries: bool = True
    ):
        """Initialize retry context.
        
        Args:
            max_attempts: Maximum number of attempts
            base_delay: Base delay for exponential backoff
            log_retries: Whether to log retry attempts
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.log_retries = log_retries
        self.current_attempt = 0
    
    def __iter__(self):
        """Iterate over retry attempts."""
        self.current_attempt = 0
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
        if self.current_attempt > 0:
            delay = calculate_backoff_delay(self.current_attempt - 1, self.base_delay)
            
            if self.log_retries:
                logger.info(f"Waiting {delay}s before retry...")
            
            time.sleep(delay)


def retry_llm_call(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs
):
    """Retry an LLM API call with exponential backoff.
    
    This is a functional interface for retry logic that doesn't require
    using a decorator.
    
    Args:
        func: Function to call
        *args: Positional arguments for function
        max_attempts: Maximum number of attempts
        base_delay: Base delay for exponential backoff
        **kwargs: Keyword arguments for function
        
    Returns:
        Result of successful function call
        
    Raises:
        Exception: If all retries are exhausted
        
    Example:
        >>> result = retry_llm_call(
        ...     provider.generate,
        ...     request,
        ...     max_attempts=3,
        ...     base_delay=1.0
        ... )
    """
    retry_ctx = RetryContext(max_attempts, base_delay)
    last_exception = None
    
    for attempt in retry_ctx:
        try:
            return func(*args, **kwargs)
        
        except Exception as e:
            if not retry_ctx.should_retry(e):
                raise
            
            last_exception = e
            retry_ctx.wait()
    
    # All retries exhausted
    if last_exception:
        raise last_exception
