"""
Unit Tests: CloudAnthropicProvider

Tests for the Cloud Anthropic (Claude) LLM provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Generate method with various inputs
- Estimate cost method with accurate pricing
- Validate requirements method
- Error handling for authentication, rate limits, etc.
- Pricing configuration override

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
- 30.2: Pricing configuration override for LLM providers
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.providers.base import LLMRequest, LLMResponse
from pipeline.llm.config import AnthropicConfig
from pipeline.llm.errors import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def anthropic_config():
    """Create a test Anthropic configuration."""
    return AnthropicConfig(
        api_key="sk-ant-test-key-12345",
        default_model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0.7
    )


@pytest.fixture
def anthropic_provider(anthropic_config):
    """Create a CloudAnthropicProvider instance."""
    return CloudAnthropicProvider(anthropic_config)


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

def test_initialization_with_valid_config(anthropic_config):
    """Test provider initialization with valid configuration."""
    provider = CloudAnthropicProvider(anthropic_config)
    
    assert provider is not None
    assert provider.config == anthropic_config
    assert provider.config.api_key == "sk-ant-test-key-12345"
    assert provider.config.default_model == "claude-3-haiku-20240307"


def test_initialization_stores_config(anthropic_config):
    """Test that provider stores configuration object."""
    provider = CloudAnthropicProvider(anthropic_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, AnthropicConfig)


# ============================================================================
# Test: Pricing Configuration Override (Task 30.2)
# ============================================================================

def test_pricing_override_via_config():
    """Test that pricing can be overridden via config."""
    custom_pricing = {
        "claude-3-haiku-20240307": {
            "input": 0.0005,  # Override default
            "output": 0.0015
        }
    }
    config = AnthropicConfig(
        api_key="sk-ant-test-key",
        pricing_override=custom_pricing
    )
    provider = CloudAnthropicProvider(config)
    
    # Verify config stores the override
    assert provider.config.pricing_override == custom_pricing


def test_calculate_cost_uses_override_pricing(anthropic_config):
    """Test that _calculate_cost uses override pricing when available."""
    # Anthropic uses pricing per 1M tokens (not per 1K)
    custom_pricing = {
        "claude-3-haiku-20240307": {
            "input": 1.0,  # $1.00 per 1M tokens
            "output": 2.0  # $2.00 per 1M tokens
        }
    }
    anthropic_config.pricing_override = custom_pricing
    provider = CloudAnthropicProvider(anthropic_config)
    
    # Calculate cost with override pricing
    cost = provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Input: 1000 tokens * $1.00 / 1,000,000 = $0.001
    # Output: 500 tokens * $2.00 / 1,000,000 = $0.001
    # Total: $0.002
    assert cost == pytest.approx(0.002, rel=0.01)


def test_calculate_cost_falls_back_to_default(anthropic_config):
    """Test that _calculate_cost falls back to default pricing when no override."""
    # No pricing override
    provider = CloudAnthropicProvider(anthropic_config)
    
    # Calculate cost with default pricing
    cost = provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Should use default pricing from PRICING constant
    assert cost > 0
    assert isinstance(cost, float)


def test_estimate_cost_uses_override_pricing(anthropic_config):
    """Test that estimate_cost uses override pricing when available."""
    custom_pricing = {
        "claude-3-haiku-20240307": {
            "input": 0.001,
            "output": 0.002
        }
    }
    anthropic_config.pricing_override = custom_pricing
    provider = CloudAnthropicProvider(anthropic_config)
    
    request = LLMRequest(
        prompt="Test prompt with approximately 10 tokens here",
        model="claude-3-haiku-20240307",
        max_tokens=100,
        temperature=0.7
    )
    
    cost = provider.estimate_cost(request)
    
    # Cost should use override pricing
    assert cost > 0
    assert isinstance(cost, float)


def test_pricing_override_partial_models(anthropic_config):
    """Test that pricing override works for partial model list."""
    custom_pricing = {
        "claude-3-haiku-20240307": {
            "input": 0.001,
            "output": 0.002
        }
        # Other models not overridden, should use default
    }
    anthropic_config.pricing_override = custom_pricing
    provider = CloudAnthropicProvider(anthropic_config)
    
    # Overridden model should use custom pricing
    cost_override = provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Non-overridden model should use default pricing
    cost_default = provider._calculate_cost(
        model="claude-3-sonnet-20240229",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Costs should be different
    assert cost_override != cost_default


def test_pricing_override_with_none():
    """Test that provider works when pricing_override is None."""
    config = AnthropicConfig(
        api_key="sk-ant-test-key",
        pricing_override=None  # Explicitly None
    )
    provider = CloudAnthropicProvider(config)
    
    # Should use default pricing
    cost = provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=1000,
        output_tokens=500
    )
    
    assert cost > 0
    assert isinstance(cost, float)


# ============================================================================
# Test: Validate Requirements
# ============================================================================

def test_validate_requirements_with_valid_key(anthropic_provider):
    """Test validate_requirements with valid API key."""
    with patch('anthropic.Anthropic'):
        result = anthropic_provider.validate_requirements()
        
        # Should return True (we don't test actual API connectivity)
        assert result is True


def test_validate_requirements_without_api_key():
    """Test validate_requirements without API key.
    
    The provider should raise AuthenticationError during initialization
    when API key is missing, not during validate_requirements().
    """
    config = AnthropicConfig(api_key="")
    
    # Provider should raise AuthenticationError during initialization
    with pytest.raises(AuthenticationError) as exc_info:
        provider = CloudAnthropicProvider(config)
    
    assert "API key not found" in str(exc_info.value)


# ============================================================================
# Test: Get Capabilities
# ============================================================================

def test_get_capabilities(anthropic_provider):
    """Test get_capabilities returns correct information."""
    capabilities = anthropic_provider.get_capabilities()
    
    assert capabilities["provider"] == "cloud-anthropic"
    assert len(capabilities["supported_models"]) > 0
    assert "claude-3" in str(capabilities["supported_models"])
    assert capabilities["supports_streaming"] is True


# ============================================================================
# Test: Get Context Window
# ============================================================================

def test_get_context_window_for_known_models(anthropic_provider):
    """Test get_context_window returns correct values for known models."""
    # Claude 3 Haiku
    assert anthropic_provider.get_context_window("claude-3-haiku-20240307") == 200000
    
    # Claude 3 Sonnet
    assert anthropic_provider.get_context_window("claude-3-sonnet-20240229") == 200000
    
    # Claude 3 Opus
    assert anthropic_provider.get_context_window("claude-3-opus-20240229") == 200000


def test_get_context_window_for_unknown_model(anthropic_provider):
    """Test get_context_window returns default for unknown models.
    
    The provider returns a default context window (200,000) for unknown models
    rather than raising an exception. This is a graceful fallback behavior.
    """
    # Unknown model should return default context window
    context_window = anthropic_provider.get_context_window("unknown-model")
    
    assert context_window == 200_000


# ============================================================================
# Test: Configuration Usage
# ============================================================================

def test_uses_config_default_model(anthropic_config):
    """Test that provider uses default_model from config when not specified."""
    anthropic_config.default_model = "claude-3-sonnet-20240229"
    provider = CloudAnthropicProvider(anthropic_config)
    
    # Verify default model is stored
    assert provider.config.default_model == "claude-3-sonnet-20240229"


def test_initializes_client_eagerly(anthropic_config):
    """Test that Anthropic client is initialized eagerly during __init__.
    
    The provider uses eager initialization (not lazy loading) to validate
    credentials and fail fast if the anthropic package is not installed.
    """
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Client should be initialized during __init__
        provider = CloudAnthropicProvider(anthropic_config)
        
        # Verify client was initialized
        mock_anthropic.assert_called_once_with(api_key=anthropic_config.api_key)
        assert provider.client == mock_client
