"""
Unit Tests: CloudOpenAIProvider

Tests for the Cloud OpenAI LLM provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Generate method with various inputs
- Estimate cost method with accurate pricing
- Validate requirements method
- Error handling for authentication, rate limits, etc.
- Token counting with tiktoken
- Context window retrieval

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.base import LLMRequest, LLMResponse
from pipeline.llm.config import OpenAIConfig
from pipeline.llm.errors import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    TimeoutError,
    NetworkError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def openai_config():
    """Create a test OpenAI configuration."""
    return OpenAIConfig(
        api_key="sk-test-key-12345",
        default_model="gpt-4-turbo",
        max_tokens=4096,
        temperature=0.7,
        timeout=60
    )


@pytest.fixture
def openai_provider(openai_config):
    """Create a CloudOpenAIProvider instance."""
    return CloudOpenAIProvider(openai_config)


@pytest.fixture
def sample_request():
    """Create a sample LLM request."""
    return LLMRequest(
        prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.7
    )


# ============================================================================
# Test: Initialization
# ============================================================================

def test_initialization_with_valid_config(openai_config):
    """Test provider initialization with valid configuration."""
    provider = CloudOpenAIProvider(openai_config)
    
    assert provider is not None
    assert provider.config == openai_config
    assert provider.config.api_key == "sk-test-key-12345"
    assert provider.config.default_model == "gpt-4-turbo"


def test_initialization_stores_config(openai_config):
    """Test that provider stores configuration object."""
    provider = CloudOpenAIProvider(openai_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, OpenAIConfig)


# ============================================================================
# Test: Generate Method
# ============================================================================

@patch('tiktoken.get_encoding')
def test_generate_with_valid_request(mock_get_encoding, openai_provider, sample_request):
    """Test generate method with valid request."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    mock_get_encoding.return_value = mock_encoding
    
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    
    # Mock successful API response with proper structure
    mock_choice = Mock()
    mock_choice.message.content = "The capital of France is Paris."
    mock_choice.finish_reason = "stop"
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.id = "chatcmpl-123"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set the mock client
    openai_provider._client = mock_client
    
    # Generate completion
    response = openai_provider.generate(sample_request)
    
    # Verify response
    assert isinstance(response, LLMResponse)
    assert response.content == "The capital of France is Paris."
    assert response.model_used == "gpt-4-turbo"
    assert response.tokens_used > 0
    assert response.cost_usd > 0  # Cloud models have cost
    assert "input_tokens" in response.metadata
    assert "output_tokens" in response.metadata


@patch('tiktoken.get_encoding')
def test_generate_uses_custom_model(mock_get_encoding, openai_provider):
    """Test generate method uses custom model from request."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
    mock_get_encoding.return_value = mock_encoding
    
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    
    # Mock successful API response with proper structure
    mock_choice = Mock()
    mock_choice.message.content = "Test response"
    mock_choice.finish_reason = "stop"
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.id = "chatcmpl-123"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set the mock client
    openai_provider._client = mock_client
    
    # Request with custom model
    request = LLMRequest(
        prompt="Test prompt",
        model="gpt-3.5-turbo",
        max_tokens=50,
        temperature=0.5
    )
    
    response = openai_provider.generate(request)
    
    # Verify custom model was used
    assert response.model_used == "gpt-3.5-turbo"


# ============================================================================
# Test: Error Handling
# ============================================================================

def test_generate_handles_rate_limit_error(openai_provider, sample_request):
    """Test generate method handles rate limit errors."""
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded (429)")
    openai_provider._client = mock_client
    
    # Should raise RateLimitError
    with pytest.raises(RateLimitError) as exc_info:
        openai_provider.generate(sample_request)
    
    assert "rate limit" in str(exc_info.value).lower()


def test_generate_handles_authentication_error(openai_provider, sample_request):
    """Test generate method handles authentication errors."""
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("Authentication failed (401)")
    openai_provider._client = mock_client
    
    # Should raise AuthenticationError
    with pytest.raises(AuthenticationError) as exc_info:
        openai_provider.generate(sample_request)
    
    assert "authentication" in str(exc_info.value).lower()


def test_generate_handles_invalid_request_error(openai_provider, sample_request):
    """Test generate method handles invalid request errors."""
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("Invalid request (400)")
    openai_provider._client = mock_client
    
    # Should raise InvalidRequestError
    with pytest.raises(InvalidRequestError) as exc_info:
        openai_provider.generate(sample_request)
    
    assert "invalid" in str(exc_info.value).lower()


# ============================================================================
# Test: Estimate Cost
# ============================================================================

def test_estimate_cost_calculates_correctly(openai_provider):
    """Test estimate_cost calculates cost based on pricing."""
    request = LLMRequest(
        prompt="This is a test prompt",
        max_tokens=100,
        temperature=0.7
    )
    
    cost = openai_provider.estimate_cost(request)
    
    # Cost should be positive for cloud models
    assert cost > 0
    assert isinstance(cost, float)


def test_estimate_cost_varies_by_model(openai_provider):
    """Test estimate_cost varies based on model pricing."""
    # GPT-4 is more expensive than GPT-3.5
    gpt4_request = LLMRequest(
        prompt="Test prompt",
        model="gpt-4",
        max_tokens=100,
        temperature=0.7
    )
    
    gpt35_request = LLMRequest(
        prompt="Test prompt",
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7
    )
    
    gpt4_cost = openai_provider.estimate_cost(gpt4_request)
    gpt35_cost = openai_provider.estimate_cost(gpt35_request)
    
    assert gpt4_cost > gpt35_cost


# ============================================================================
# Test: Validate Requirements
# ============================================================================

@patch('pipeline.llm.providers.cloud_openai.CloudOpenAIProvider.client', new_callable=lambda: Mock())
def test_validate_requirements_with_valid_key(mock_client_prop, openai_provider):
    """Test validate_requirements returns True with valid API key."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_client_prop.return_value = mock_client
    openai_provider._client = mock_client
    
    # Mock successful models.list() call
    mock_client.models.list.return_value = []
    
    result = openai_provider.validate_requirements()
    
    assert result is True


def test_validate_requirements_without_api_key():
    """Test validate_requirements returns False without API key."""
    config = OpenAIConfig(api_key="")
    provider = CloudOpenAIProvider(config)
    
    result = provider.validate_requirements()
    
    assert result is False


# ============================================================================
# Test: Get Capabilities
# ============================================================================

def test_get_capabilities(openai_provider):
    """Test get_capabilities returns correct information."""
    capabilities = openai_provider.get_capabilities()
    
    assert capabilities["provider"] == "cloud-openai"
    assert "gpt-4" in capabilities["supported_models"]
    assert "gpt-3.5-turbo" in capabilities["supported_models"]
    assert capabilities["supports_streaming"] is True
    assert capabilities["supports_functions"] is True


# ============================================================================
# Test: Get Context Window
# ============================================================================

def test_get_context_window_for_known_models(openai_provider):
    """Test get_context_window returns correct values for known models."""
    assert openai_provider.get_context_window("gpt-3.5-turbo") == 16385
    assert openai_provider.get_context_window("gpt-4-turbo") == 128000
    assert openai_provider.get_context_window("gpt-4") == 8192


def test_get_context_window_raises_for_unknown_model(openai_provider):
    """Test get_context_window raises ValueError for unknown models."""
    with pytest.raises(ValueError) as exc_info:
        openai_provider.get_context_window("unknown-model")
    
    assert "not supported" in str(exc_info.value)


# ============================================================================
# Test: Token Counting
# ============================================================================

@patch('tiktoken.get_encoding')
def test_count_tokens_uses_tiktoken(mock_get_encoding, openai_provider):
    """Test token counting uses tiktoken."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    mock_get_encoding.return_value = mock_encoding
    
    count = openai_provider._count_tokens("Test text", "gpt-4")
    
    assert count == 5
    mock_get_encoding.assert_called_once_with("cl100k_base")


# ============================================================================
# Test: Configuration Usage
# ============================================================================

@patch('tiktoken.get_encoding')
def test_uses_config_default_model(mock_get_encoding, openai_config):
    """Test that provider uses default_model from config when not specified."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2]  # 2 tokens
    mock_get_encoding.return_value = mock_encoding
    
    # Create provider with custom default model
    openai_config.default_model = "gpt-3.5-turbo"
    provider = CloudOpenAIProvider(openai_config)
    
    # Mock OpenAI client by setting _client directly
    mock_client = Mock()
    
    # Mock successful API response with proper structure
    mock_choice = Mock()
    mock_choice.message.content = "Test"
    mock_choice.finish_reason = "stop"
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.id = "chatcmpl-123"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set the mock client
    provider._client = mock_client
    
    # Request without model specified
    request = LLMRequest(prompt="Test", max_tokens=50, temperature=0.5)
    response = provider.generate(request)
    
    # Verify default model was used
    assert response.model_used == "gpt-3.5-turbo"


def test_lazy_loads_client(openai_config):
    """Test that OpenAI client is lazy-loaded."""
    provider = CloudOpenAIProvider(openai_config)
    
    # Client should not be initialized yet
    assert provider._client is None
    
    # Accessing client property should initialize it
    with patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value = Mock()
        client = provider.client
        assert client is not None
        mock_openai.assert_called_once()


# ============================================================================
# Test: Pricing Configuration Override (Task 30.2)
# ============================================================================

def test_pricing_override_via_config():
    """Test that pricing can be overridden via config."""
    custom_pricing = {
        "gpt-4": {
            "input_per_1k": 0.05,  # Override default
            "output_per_1k": 0.10
        }
    }
    config = OpenAIConfig(
        api_key="sk-test-key",
        pricing_override=custom_pricing
    )
    provider = CloudOpenAIProvider(config)
    
    # Verify config stores the override
    assert provider.config.pricing_override == custom_pricing


@patch('tiktoken.get_encoding')
def test_get_pricing_returns_override_when_available(mock_get_encoding, openai_config):
    """Test that _get_pricing returns override pricing when available."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3]
    mock_get_encoding.return_value = mock_encoding
    
    custom_pricing = {
        "gpt-4": {
            "input_per_1k": 0.05,
            "output_per_1k": 0.10
        }
    }
    openai_config.pricing_override = custom_pricing
    provider = CloudOpenAIProvider(openai_config)
    
    # Get pricing for overridden model
    pricing = provider._get_pricing("gpt-4")
    
    assert pricing == custom_pricing["gpt-4"]
    assert pricing["input_per_1k"] == 0.05
    assert pricing["output_per_1k"] == 0.10


@patch('tiktoken.get_encoding')
def test_get_pricing_falls_back_to_default(mock_get_encoding, openai_config):
    """Test that _get_pricing falls back to default pricing when no override."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3]
    mock_get_encoding.return_value = mock_encoding
    
    # No pricing override
    provider = CloudOpenAIProvider(openai_config)
    
    # Get pricing for model without override
    pricing = provider._get_pricing("gpt-4")
    
    # Should return default pricing from PRICING constant
    assert pricing is not None
    assert "input_per_1k" in pricing
    assert "output_per_1k" in pricing


@patch('tiktoken.get_encoding')
def test_estimate_cost_uses_override_pricing(mock_get_encoding, openai_config):
    """Test that estimate_cost uses override pricing when available."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1] * 100  # 100 tokens
    mock_get_encoding.return_value = mock_encoding
    
    custom_pricing = {
        "gpt-4": {
            "input_per_1k": 0.10,  # $0.10 per 1K tokens (much higher than default)
            "output_per_1k": 0.20
        }
    }
    openai_config.pricing_override = custom_pricing
    provider = CloudOpenAIProvider(openai_config)
    
    request = LLMRequest(
        prompt="Test prompt",
        model="gpt-4",
        max_tokens=100,
        temperature=0.7
    )
    
    cost = provider.estimate_cost(request)
    
    # Cost should use override pricing
    # Input: 100 tokens * $0.10 / 1000 = $0.01
    # Output: 100 tokens * $0.20 / 1000 = $0.02
    # Total: $0.03
    assert cost == pytest.approx(0.03, rel=0.01)


@patch('tiktoken.get_encoding')
def test_generate_uses_override_pricing_for_cost_calculation(mock_get_encoding, openai_config):
    """Test that generate uses override pricing for cost calculation."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1] * 50  # 50 tokens
    mock_get_encoding.return_value = mock_encoding
    
    custom_pricing = {
        "gpt-4": {
            "input_per_1k": 0.10,
            "output_per_1k": 0.20
        }
    }
    openai_config.pricing_override = custom_pricing
    provider = CloudOpenAIProvider(openai_config)
    
    # Mock OpenAI client
    mock_client = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "Test response"
    mock_choice.finish_reason = "stop"
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.id = "chatcmpl-123"
    mock_client.chat.completions.create.return_value = mock_response
    provider._client = mock_client
    
    request = LLMRequest(
        prompt="Test",
        model="gpt-4",
        max_tokens=50,
        temperature=0.7
    )
    
    response = provider.generate(request)
    
    # Verify cost was calculated using override pricing
    # Input: 50 tokens * $0.10 / 1000 = $0.005
    # Output: 50 tokens * $0.20 / 1000 = $0.010
    # Total: $0.015
    assert response.cost_usd == pytest.approx(0.015, rel=0.01)


def test_pricing_override_partial_models():
    """Test that pricing override works for partial model list."""
    custom_pricing = {
        "gpt-4": {
            "input_per_1k": 0.05,
            "output_per_1k": 0.10
        }
        # gpt-3.5-turbo not overridden, should use default
    }
    config = OpenAIConfig(
        api_key="sk-test-key",
        pricing_override=custom_pricing
    )
    provider = CloudOpenAIProvider(config)
    
    # Overridden model should use custom pricing
    gpt4_pricing = provider._get_pricing("gpt-4")
    assert gpt4_pricing["input_per_1k"] == 0.05
    
    # Non-overridden model should use default pricing
    gpt35_pricing = provider._get_pricing("gpt-3.5-turbo")
    assert gpt35_pricing["input_per_1k"] != 0.05  # Should be default value
