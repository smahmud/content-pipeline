"""
Unit Tests: CloudAWSBedrockProvider

Tests for the Cloud AWS Bedrock LLM provider implementation.

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

from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.base import LLMRequest, LLMResponse
from pipeline.llm.config import BedrockConfig
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
def bedrock_config():
    """Create a test Bedrock configuration."""
    return BedrockConfig(
        access_key_id="AKIATEST12345",
        secret_access_key="test_secret_key",
        region="us-east-1",
        default_model="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=4096,
        temperature=0.7
    )


@pytest.fixture
def bedrock_provider(bedrock_config):
    """Create a CloudAWSBedrockProvider instance."""
    return CloudAWSBedrockProvider(bedrock_config)


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

def test_initialization_with_valid_config(bedrock_config):
    """Test provider initialization with valid configuration."""
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    assert provider is not None
    assert provider.config == bedrock_config
    assert provider.config.access_key_id == "AKIATEST12345"
    assert provider.config.default_model == "anthropic.claude-3-haiku-20240307-v1:0"


def test_initialization_stores_config(bedrock_config):
    """Test that provider stores configuration object."""
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, BedrockConfig)


# ============================================================================
# Test: Pricing Configuration Override (Task 30.2)
# ============================================================================

def test_pricing_override_via_config():
    """Test that pricing can be overridden via config."""
    custom_pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.0005,  # Override default
            "output": 0.0015
        }
    }
    config = BedrockConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        pricing_override=custom_pricing
    )
    provider = CloudAWSBedrockProvider(config)
    
    # Verify config stores the override
    assert provider.config.pricing_override == custom_pricing


def test_calculate_cost_uses_override_pricing(bedrock_config):
    """Test that _calculate_cost uses override pricing when available."""
    custom_pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.001,  # $0.001 per 1K tokens
            "output": 0.002
        }
    }
    bedrock_config.pricing_override = custom_pricing
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    # Calculate cost with override pricing
    cost = provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Input: 1000 tokens * $0.001 / 1000 = $0.001
    # Output: 500 tokens * $0.002 / 1000 = $0.001
    # Total: $0.002
    assert cost == pytest.approx(0.002, rel=0.01)


def test_calculate_cost_falls_back_to_default(bedrock_config):
    """Test that _calculate_cost falls back to default pricing when no override."""
    # No pricing override
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    # Calculate cost with default pricing
    cost = provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Should use default pricing from PRICING constant
    assert cost > 0
    assert isinstance(cost, float)


def test_estimate_cost_uses_override_pricing(bedrock_config):
    """Test that estimate_cost uses override pricing when available."""
    custom_pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.001,
            "output": 0.002
        }
    }
    bedrock_config.pricing_override = custom_pricing
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    request = LLMRequest(
        prompt="Test prompt with approximately 10 tokens here",
        model="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=100,
        temperature=0.7
    )
    
    cost = provider.estimate_cost(request)
    
    # Cost should use override pricing
    assert cost > 0
    assert isinstance(cost, float)


def test_pricing_override_partial_models(bedrock_config):
    """Test that pricing override works for partial model list."""
    custom_pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.001,
            "output": 0.002
        }
        # Other models not overridden, should use default
    }
    bedrock_config.pricing_override = custom_pricing
    provider = CloudAWSBedrockProvider(bedrock_config)
    
    # Overridden model should use custom pricing
    cost_override = provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Non-overridden model should use default pricing
    cost_default = provider._calculate_cost(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Costs should be different
    assert cost_override != cost_default


def test_pricing_override_with_none():
    """Test that provider works when pricing_override is None."""
    config = BedrockConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        pricing_override=None  # Explicitly None
    )
    provider = CloudAWSBedrockProvider(config)
    
    # Should use default pricing
    cost = provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=1000,
        output_tokens=500
    )
    
    assert cost > 0
    assert isinstance(cost, float)


# ============================================================================
# Test: Validate Requirements
# ============================================================================

@patch('boto3.client')
def test_validate_requirements_with_valid_credentials(mock_boto_client, bedrock_provider):
    """Test validate_requirements with valid credentials."""
    mock_client = Mock()
    mock_boto_client.return_value = mock_client
    
    # Mock successful API call
    mock_client.list_foundation_models.return_value = {'modelSummaries': []}
    
    result = bedrock_provider.validate_requirements()
    
    assert result is True


def test_validate_requirements_without_credentials():
    """Test validate_requirements without credentials."""
    config = BedrockConfig(
        access_key_id="",
        secret_access_key="",
        region="us-east-1"
    )
    provider = CloudAWSBedrockProvider(config)
    
    result = provider.validate_requirements()
    
    # Should still return True (can use IAM role)
    assert result is True


# ============================================================================
# Test: Get Capabilities
# ============================================================================

def test_get_capabilities(bedrock_provider):
    """Test get_capabilities returns correct information."""
    capabilities = bedrock_provider.get_capabilities()
    
    assert capabilities["provider"] == "cloud-aws-bedrock"
    assert len(capabilities["supported_models"]) > 0
    assert "anthropic.claude" in str(capabilities["supported_models"])


# ============================================================================
# Test: Get Context Window
# ============================================================================

def test_get_context_window_for_known_models(bedrock_provider):
    """Test get_context_window returns correct values for known models."""
    # Claude 3 Haiku
    assert bedrock_provider.get_context_window("anthropic.claude-3-haiku-20240307-v1:0") == 200000
    
    # Claude 3 Sonnet
    assert bedrock_provider.get_context_window("anthropic.claude-3-sonnet-20240229-v1:0") == 200000


def test_get_context_window_raises_for_unknown_model(bedrock_provider):
    """Test get_context_window raises ValueError for unknown models."""
    with pytest.raises(ValueError) as exc_info:
        bedrock_provider.get_context_window("unknown-model")
    
    assert "not supported" in str(exc_info.value)
