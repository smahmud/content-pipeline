"""
Unit tests for Retry Logic

Tests retry decorators, exponential backoff, and error classification.
"""

import pytest
import time
from unittest.mock import Mock, patch

from pipeline.enrichment.retry import (
    is_transient_error,
    is_permanent_error,
    calculate_backoff_delay,
    retry_with_backoff,
    RetryContext,
    retry_llm_call
)
from pipeline.enrichment.errors import (
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    InvalidRequestError
)


class TestErrorClassification:
    """Test suite for error classification."""
    
    def test_is_transient_error_rate_limit(self):
        """Test rate limit error is transient."""
        error = RateLimitError("Rate limit exceeded")
        
        assert is_transient_error(error) is True
        assert is_permanent_error(error) is False
    
    def test_is_transient_error_timeout(self):
        """Test timeout error is transient."""
        error = TimeoutError("Request timed out")
        
        assert is_transient_error(error) is True
        assert is_permanent_error(error) is False
    
    def test_is_transient_error_network(self):
        """Test network error is transient."""
        error = NetworkError("Connection failed")
        
        assert is_transient_error(error) is True
        assert is_permanent_error(error) is False
    
    def test_is_permanent_error_authentication(self):
        """Test authentication error is permanent."""
        error = AuthenticationError("Invalid API key")
        
        assert is_permanent_error(error) is True
        assert is_transient_error(error) is False
    
    def test_is_permanent_error_invalid_request(self):
        """Test invalid request error is permanent."""
        error = InvalidRequestError("Invalid parameters")
        
        assert is_permanent_error(error) is True
        assert is_transient_error(error) is False
    
    def test_is_transient_error_unknown(self):
        """Test unknown error is not transient."""
        error = ValueError("Some error")
        
        assert is_transient_error(error) is False
        assert is_permanent_error(error) is False


class TestBackoffCalculation:
    """Test suite for backoff delay calculation."""
    
    def test_calculate_backoff_delay_attempt_0(self):
        """Test backoff delay for first retry."""
        delay = calculate_backoff_delay(0, base_delay=1.0)
        
        assert delay == 1.0
    
    def test_calculate_backoff_delay_attempt_1(self):
        """Test backoff delay for second retry."""
        delay = calculate_backoff_delay(1, base_delay=1.0)
        
        assert delay == 2.0
    
    def test_calculate_backoff_delay_attempt_2(self):
        """Test backoff delay for third retry."""
        delay = calculate_backoff_delay(2, base_delay=1.0)
        
        assert delay == 4.0
    
    def test_calculate_backoff_delay_custom_base(self):
        """Test backoff delay with custom base delay."""
        delay = calculate_backoff_delay(1, base_delay=2.0)
        
        assert delay == 4.0


class TestRetryDecorator:
    """Test suite for retry_with_backoff decorator."""
    
    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        mock_func = Mock(return_value="success")
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_transient_error_then_success(self):
        """Test retry after transient error."""
        mock_func = Mock(side_effect=[
            RateLimitError("Rate limit"),
            "success"
        ])
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3, base_delay=0.01)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_permanent_error_no_retry(self):
        """Test permanent error is not retried."""
        mock_func = Mock(side_effect=AuthenticationError("Invalid key"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3)(mock_func)
        
        with pytest.raises(AuthenticationError):
            decorated()
        
        # Should only be called once (no retry)
        assert mock_func.call_count == 1
    
    def test_retry_exhausted_attempts(self):
        """Test all retry attempts are exhausted."""
        mock_func = Mock(side_effect=RateLimitError("Rate limit"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3, base_delay=0.01)(mock_func)
        
        with pytest.raises(RateLimitError):
            decorated()
        
        # Should be called max_attempts times
        assert mock_func.call_count == 3
    
    def test_retry_unknown_error_no_retry(self):
        """Test unknown error is not retried."""
        mock_func = Mock(side_effect=ValueError("Unknown error"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3)(mock_func)
        
        with pytest.raises(ValueError):
            decorated()
        
        # Should only be called once
        assert mock_func.call_count == 1
    
    @patch('time.sleep')
    def test_retry_backoff_delays(self, mock_sleep):
        """Test exponential backoff delays."""
        mock_func = Mock(side_effect=[
            RateLimitError("Rate limit"),
            RateLimitError("Rate limit"),
            "success"
        ])
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3, base_delay=1.0)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        
        # Verify sleep was called with correct delays
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1.0  # First retry: 1s
        assert mock_sleep.call_args_list[1][0][0] == 2.0  # Second retry: 2s
    
    def test_retry_with_args_and_kwargs(self):
        """Test retry with function arguments."""
        mock_func = Mock(return_value="success")
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3)(mock_func)
        
        result = decorated("arg1", "arg2", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")


class TestRetryContext:
    """Test suite for RetryContext."""
    
    def test_retry_context_iteration(self):
        """Test retry context iteration."""
        ctx = RetryContext(max_attempts=3)
        
        attempts = list(ctx)
        
        assert attempts == [0, 1, 2]
    
    def test_retry_context_should_retry_transient(self):
        """Test should_retry for transient error."""
        ctx = RetryContext(max_attempts=3)
        next(iter(ctx))  # Start iteration
        
        should_retry = ctx.should_retry(RateLimitError("Rate limit"))
        
        assert should_retry is True
    
    def test_retry_context_should_retry_permanent(self):
        """Test should_retry for permanent error."""
        ctx = RetryContext(max_attempts=3)
        next(iter(ctx))  # Start iteration
        
        should_retry = ctx.should_retry(AuthenticationError("Invalid key"))
        
        assert should_retry is False
    
    def test_retry_context_should_retry_exhausted(self):
        """Test should_retry when attempts exhausted."""
        ctx = RetryContext(max_attempts=2)
        
        # Exhaust attempts
        for _ in ctx:
            pass
        
        should_retry = ctx.should_retry(RateLimitError("Rate limit"))
        
        assert should_retry is False
    
    @patch('time.sleep')
    def test_retry_context_wait(self, mock_sleep):
        """Test retry context wait."""
        ctx = RetryContext(max_attempts=3, base_delay=1.0)
        
        # First attempt
        next(iter(ctx))
        ctx.wait()
        
        # After first attempt, current_attempt is 1, so it should sleep with 1s delay
        mock_sleep.assert_called_once_with(1.0)
        
        # Second attempt
        next(ctx)
        ctx.wait()
        
        # After second attempt, should sleep with 2s delay
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[1][0][0] == 2.0
    
    def test_retry_context_usage_pattern(self):
        """Test typical retry context usage pattern."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limit")
            return "success"
        
        ctx = RetryContext(max_attempts=3, base_delay=0.01)
        result = None
        
        for attempt in ctx:
            try:
                result = failing_func()
                break
            except RateLimitError as e:
                if not ctx.should_retry(e):
                    raise
                ctx.wait()
        
        assert result == "success"
        assert call_count == 3


class TestRetryLLMCall:
    """Test suite for retry_llm_call function."""
    
    def test_retry_llm_call_success(self):
        """Test successful LLM call."""
        mock_func = Mock(return_value="success")
        
        result = retry_llm_call(mock_func, "arg1", max_attempts=3, base_delay=0.01)
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1")
    
    def test_retry_llm_call_with_retry(self):
        """Test LLM call with retry."""
        mock_func = Mock(side_effect=[
            RateLimitError("Rate limit"),
            "success"
        ])
        
        result = retry_llm_call(mock_func, max_attempts=3, base_delay=0.01)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_llm_call_permanent_error(self):
        """Test LLM call with permanent error."""
        mock_func = Mock(side_effect=AuthenticationError("Invalid key"))
        
        with pytest.raises(AuthenticationError):
            retry_llm_call(mock_func, max_attempts=3, base_delay=0.01)
        
        assert mock_func.call_count == 1
    
    def test_retry_llm_call_exhausted(self):
        """Test LLM call with exhausted retries."""
        mock_func = Mock(side_effect=RateLimitError("Rate limit"))
        
        with pytest.raises(RateLimitError):
            retry_llm_call(mock_func, max_attempts=3, base_delay=0.01)
        
        assert mock_func.call_count == 3
    
    def test_retry_llm_call_with_kwargs(self):
        """Test LLM call with keyword arguments."""
        mock_func = Mock(return_value="success")
        
        result = retry_llm_call(
            mock_func,
            "arg1",
            max_attempts=3,
            base_delay=0.01,
            kwarg1="value1"
        )
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")


class TestRetryLogging:
    """Test suite for retry logging."""
    
    @patch('pipeline.enrichment.retry.logger')
    def test_retry_logs_transient_error(self, mock_logger):
        """Test that transient errors are logged."""
        mock_func = Mock(side_effect=[
            RateLimitError("Rate limit"),
            "success"
        ])
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3, base_delay=0.01, log_retries=True)(mock_func)
        
        decorated()
        
        # Verify warning was logged
        assert mock_logger.warning.called
    
    @patch('pipeline.enrichment.retry.logger')
    def test_retry_logs_permanent_error(self, mock_logger):
        """Test that permanent errors are logged."""
        mock_func = Mock(side_effect=AuthenticationError("Invalid key"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=3, log_retries=True)(mock_func)
        
        with pytest.raises(AuthenticationError):
            decorated()
        
        # Verify error was logged
        assert mock_logger.error.called
    
    @patch('pipeline.enrichment.retry.logger')
    def test_retry_logs_exhausted_attempts(self, mock_logger):
        """Test that exhausted attempts are logged."""
        mock_func = Mock(side_effect=RateLimitError("Rate limit"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_attempts=2, base_delay=0.01, log_retries=True)(mock_func)
        
        with pytest.raises(RateLimitError):
            decorated()
        
        # Verify error was logged for exhausted attempts
        assert mock_logger.error.called
