"""
Property-Based Tests: Pricing Configuration

Tests that verify pricing configuration override functionality across all providers.

**Test Coverage:**
- All transcription providers respect cost_per_minute_usd override
- All LLM providers respect pricing_override
- Pricing overrides don't break cost estimation
- Default pricing is used when no override is provided

**Requirements Validated:**
- 30.1: Pricing configuration for transcription providers
- 30.2: Pricing configuration for LLM providers
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch

from pipeline.transcription.config import WhisperAPIConfig, AWSTranscribeConfig
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider

from pipeline.llm.config import OpenAIConfig, BedrockConfig, AnthropicConfig
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.providers.base import LLMRequest


# ============================================================================
# Property Tests: Transcription Provider Pricing
# ============================================================================

@given(
    cost_per_minute=st.floats(min_value=0.001, max_value=1.0),
    duration_seconds=st.floats(min_value=0.0, max_value=3600.0)
)
def test_whisper_api_respects_cost_override(cost_per_minute, duration_seconds):
    """Property: WhisperAPI provider always uses cost_per_minute_usd from config."""
    config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=cost_per_minute
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    # Calculate expected cost
    duration_minutes = duration_seconds / 60.0
    expected_cost = round(duration_minutes * cost_per_minute, 4)
    
    # Get actual cost from provider
    actual_cost = provider.estimate_cost(duration_seconds)
    
    # Verify provider uses config value
    assert actual_cost == pytest.approx(expected_cost, rel=0.01)


@given(
    cost_per_minute=st.floats(min_value=0.001, max_value=1.0),
    duration_seconds=st.floats(min_value=0.0, max_value=3600.0)
)
def test_aws_transcribe_respects_cost_override(cost_per_minute, duration_seconds):
    """Property: AWS Transcribe provider always uses cost_per_minute_usd from config."""
    config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket",
        cost_per_minute_usd=cost_per_minute
    )
    provider = CloudAWSTranscribeProvider(config)
    
    # Calculate expected cost
    duration_minutes = duration_seconds / 60.0
    expected_cost = round(duration_minutes * cost_per_minute, 4)
    
    # Get actual cost from provider
    actual_cost = provider.estimate_cost(duration_seconds)
    
    # Verify provider uses config value
    assert actual_cost == pytest.approx(expected_cost, rel=0.01)


def test_transcription_providers_use_default_when_no_override():
    """Property: Transcription providers use default pricing when no override."""
    # WhisperAPI default
    whisper_config = WhisperAPIConfig(api_key="sk-test-key")
    whisper_provider = CloudOpenAIWhisperProvider(whisper_config)
    assert whisper_provider.config.cost_per_minute_usd == 0.006
    
    # AWS Transcribe default
    aws_config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket"
    )
    aws_provider = CloudAWSTranscribeProvider(aws_config)
    assert aws_provider.config.cost_per_minute_usd == 0.024


# ============================================================================
# Property Tests: LLM Provider Pricing
# ============================================================================

@given(
    input_price=st.floats(min_value=0.0001, max_value=0.1),
    output_price=st.floats(min_value=0.0001, max_value=0.1)
)
@patch('tiktoken.get_encoding')
def test_openai_respects_pricing_override(mock_get_encoding, input_price, output_price):
    """Property: OpenAI provider always uses pricing_override from config."""
    # Mock tiktoken encoding
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1] * 100  # 100 tokens
    mock_get_encoding.return_value = mock_encoding
    
    custom_pricing = {
        "gpt-4": {
            "input": input_price,
            "output": output_price
        }
    }
    config = OpenAIConfig(
        api_key="sk-test-key",
        pricing_override=custom_pricing
    )
    provider = CloudOpenAIProvider(config)
    
    # Get pricing for overridden model
    pricing = provider._get_pricing("gpt-4")
    
    # Verify provider uses override values
    assert pricing["input"] == input_price
    assert pricing["output"] == output_price


@given(
    input_price=st.floats(min_value=0.0001, max_value=0.1),
    output_price=st.floats(min_value=0.0001, max_value=0.1),
    input_tokens=st.integers(min_value=1, max_value=10000),
    output_tokens=st.integers(min_value=1, max_value=10000)
)
def test_bedrock_respects_pricing_override(input_price, output_price, input_tokens, output_tokens):
    """Property: Bedrock provider always uses pricing_override from config."""
    custom_pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": input_price,
            "output": output_price
        }
    }
    config = BedrockConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        pricing_override=custom_pricing
    )
    provider = CloudAWSBedrockProvider(config)
    
    # Calculate expected cost
    expected_cost = (input_tokens * input_price / 1000) + (output_tokens * output_price / 1000)
    
    # Get actual cost from provider
    actual_cost = provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    
    # Verify provider uses override values
    assert actual_cost == pytest.approx(expected_cost, rel=0.01)


@given(
    input_price=st.floats(min_value=0.0001, max_value=0.1),
    output_price=st.floats(min_value=0.0001, max_value=0.1),
    input_tokens=st.integers(min_value=1, max_value=10000),
    output_tokens=st.integers(min_value=1, max_value=10000)
)
def test_anthropic_respects_pricing_override(input_price, output_price, input_tokens, output_tokens):
    """Property: Anthropic provider always uses pricing_override from config."""
    custom_pricing = {
        "claude-3-haiku-20240307": {
            "input": input_price,
            "output": output_price
        }
    }
    config = AnthropicConfig(
        api_key="sk-ant-test-key",
        pricing_override=custom_pricing
    )
    provider = CloudAnthropicProvider(config)
    
    # Calculate expected cost
    expected_cost = (input_tokens * input_price / 1000) + (output_tokens * output_price / 1000)
    
    # Get actual cost from provider
    actual_cost = provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    
    # Verify provider uses override values
    assert actual_cost == pytest.approx(expected_cost, rel=0.01)


def test_llm_providers_use_default_when_no_override():
    """Property: LLM providers use default pricing when no override."""
    # OpenAI - no override
    openai_config = OpenAIConfig(api_key="sk-test-key")
    openai_provider = CloudOpenAIProvider(openai_config)
    assert openai_provider.config.pricing_override is None
    
    # Bedrock - no override
    bedrock_config = BedrockConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1"
    )
    bedrock_provider = CloudAWSBedrockProvider(bedrock_config)
    assert bedrock_provider.config.pricing_override is None
    
    # Anthropic - no override
    anthropic_config = AnthropicConfig(api_key="sk-ant-test-key")
    anthropic_provider = CloudAnthropicProvider(anthropic_config)
    assert anthropic_provider.config.pricing_override is None


# ============================================================================
# Property Tests: Cost Estimation Invariants
# ============================================================================

@given(
    cost_per_minute=st.floats(min_value=0.001, max_value=1.0),
    duration_seconds=st.floats(min_value=0.0, max_value=3600.0)
)
def test_transcription_cost_never_negative(cost_per_minute, duration_seconds):
    """Property: Transcription cost estimation never returns negative values."""
    # Test WhisperAPI
    whisper_config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=cost_per_minute
    )
    whisper_provider = CloudOpenAIWhisperProvider(whisper_config)
    whisper_cost = whisper_provider.estimate_cost(duration_seconds)
    assert whisper_cost >= 0
    
    # Test AWS Transcribe
    aws_config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket",
        cost_per_minute_usd=cost_per_minute
    )
    aws_provider = CloudAWSTranscribeProvider(aws_config)
    aws_cost = aws_provider.estimate_cost(duration_seconds)
    assert aws_cost >= 0


@given(
    input_price=st.floats(min_value=0.0001, max_value=0.1),
    output_price=st.floats(min_value=0.0001, max_value=0.1),
    input_tokens=st.integers(min_value=0, max_value=10000),
    output_tokens=st.integers(min_value=0, max_value=10000)
)
def test_llm_cost_never_negative(input_price, output_price, input_tokens, output_tokens):
    """Property: LLM cost estimation never returns negative values."""
    # Test Bedrock
    bedrock_config = BedrockConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        pricing_override={
            "anthropic.claude-3-haiku-20240307-v1:0": {
                "input": input_price,
                "output": output_price
            }
        }
    )
    bedrock_provider = CloudAWSBedrockProvider(bedrock_config)
    bedrock_cost = bedrock_provider._calculate_cost(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    assert bedrock_cost >= 0
    
    # Test Anthropic
    anthropic_config = AnthropicConfig(
        api_key="sk-ant-test-key",
        pricing_override={
            "claude-3-haiku-20240307": {
                "input": input_price,
                "output": output_price
            }
        }
    )
    anthropic_provider = CloudAnthropicProvider(anthropic_config)
    anthropic_cost = anthropic_provider._calculate_cost(
        model="claude-3-haiku-20240307",
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    assert anthropic_cost >= 0


@given(
    cost_per_minute=st.floats(min_value=0.001, max_value=1.0),
    duration1=st.floats(min_value=0.0, max_value=1800.0),
    duration2=st.floats(min_value=0.0, max_value=1800.0)
)
def test_transcription_cost_scales_linearly(cost_per_minute, duration1, duration2):
    """Property: Transcription cost scales linearly with duration."""
    assume(duration1 > 0 and duration2 > 0)
    assume(abs(duration1 - duration2) > 1.0)  # Ensure meaningful difference
    
    config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=cost_per_minute
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    cost1 = provider.estimate_cost(duration1)
    cost2 = provider.estimate_cost(duration2)
    
    # Cost ratio should equal duration ratio
    if cost1 > 0 and cost2 > 0:
        cost_ratio = cost1 / cost2
        duration_ratio = duration1 / duration2
        assert cost_ratio == pytest.approx(duration_ratio, rel=0.01)
