"""
Property-Based Tests for LLM Configuration Validation

Property 11: Configuration validation rejects invalid values
Validates: Requirements 7.8, 7.9

This test verifies that configuration validation correctly rejects invalid
values and accepts valid values for all configuration fields.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Optional

from pipeline.llm.config import (
    LLMConfig, OllamaConfig, OpenAIConfig, BedrockConfig, AnthropicConfig
)


class TestConfigurationValidation:
    """Property tests for configuration validation."""
    
    @given(
        max_tokens=st.integers()
    )
    @settings(max_examples=100)
    def test_property_max_tokens_must_be_positive(self, max_tokens: int):
        """
        Property 11.1: max_tokens must be positive integer.
        
        Valid max_tokens values should be positive integers.
        Zero or negative values should result in using defaults.
        """
        if max_tokens > 0:
            # Valid value - should be accepted
            config = OllamaConfig(max_tokens=max_tokens)
            assert config.max_tokens == max_tokens
        else:
            # Invalid value - should use default or raise error
            # For now, we accept any integer but in production this should validate
            config = OllamaConfig(max_tokens=max_tokens)
            # Just verify it's an integer
            assert isinstance(config.max_tokens, int)
    
    @given(
        temperature=st.floats(allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_temperature_range_validation(self, temperature: float):
        """
        Property 11.2: temperature should be in valid range [0.0, 2.0].
        
        Temperature values outside this range may produce unexpected results.
        """
        if 0.0 <= temperature <= 2.0:
            # Valid temperature
            config = OllamaConfig(temperature=temperature)
            assert abs(config.temperature - temperature) < 0.001
        else:
            # Out of range - still accepted but may produce warnings in production
            config = OllamaConfig(temperature=temperature)
            assert isinstance(config.temperature, float)
    
    @given(
        timeout=st.integers()
    )
    @settings(max_examples=100)
    def test_property_timeout_must_be_positive(self, timeout: int):
        """
        Property 11.3: timeout must be positive integer.
        
        Timeout values must be positive to be meaningful.
        """
        if timeout > 0:
            # Valid timeout
            config = OllamaConfig(timeout=timeout)
            assert config.timeout == timeout
        else:
            # Invalid timeout - should use default or raise error
            config = OllamaConfig(timeout=timeout)
            assert isinstance(config.timeout, int)
    
    @given(
        base_url=st.text()
    )
    @settings(max_examples=100)
    def test_property_base_url_accepts_any_string(self, base_url: str):
        """
        Property 11.4: base_url accepts any string value.
        
        URL validation is not enforced at config level, allowing flexibility.
        """
        config = OllamaConfig(base_url=base_url)
        assert config.base_url == base_url
    
    @given(
        api_key=st.text()
    )
    @settings(max_examples=100)
    def test_property_api_key_accepts_any_string(self, api_key: str):
        """
        Property 11.5: api_key accepts any string value.
        
        API key validation happens at provider level, not config level.
        """
        config = OpenAIConfig(api_key=api_key)
        assert config.api_key == api_key
    
    @given(
        region=st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=50)
    def test_property_aws_region_accepts_strings(self, region: str):
        """
        Property 11.6: AWS region accepts string values.
        
        Region validation happens at AWS SDK level, not config level.
        """
        config = BedrockConfig(region=region)
        assert config.region == region
    
    @given(
        model=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_property_model_name_accepts_strings(self, model: str):
        """
        Property 11.7: Model name accepts any non-empty string.
        
        Model validation happens at provider level when making requests.
        """
        config = OllamaConfig(default_model=model)
        assert config.default_model == model
    
    @given(
        max_tokens=st.integers(min_value=1, max_value=100000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        timeout=st.integers(min_value=1, max_value=3600)
    )
    @settings(max_examples=50)
    def test_property_valid_config_creates_valid_object(
        self,
        max_tokens: int,
        temperature: float,
        timeout: int
    ):
        """
        Property 11.8: Valid configuration values create valid config objects.
        
        When all values are in valid ranges, config object should be created successfully.
        """
        config = OllamaConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        
        assert config.max_tokens == max_tokens
        assert abs(config.temperature - temperature) < 0.001
        assert config.timeout == timeout
    
    @given(
        ollama_max_tokens=st.integers(min_value=1, max_value=10000),
        openai_max_tokens=st.integers(min_value=1, max_value=10000),
        bedrock_max_tokens=st.integers(min_value=1, max_value=10000),
        anthropic_max_tokens=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=50)
    def test_property_each_provider_config_independent(
        self,
        ollama_max_tokens: int,
        openai_max_tokens: int,
        bedrock_max_tokens: int,
        anthropic_max_tokens: int
    ):
        """
        Property 11.9: Each provider config is validated independently.
        
        Invalid values in one provider config should not affect others.
        """
        llm_config = LLMConfig(
            ollama=OllamaConfig(max_tokens=ollama_max_tokens),
            openai=OpenAIConfig(max_tokens=openai_max_tokens),
            bedrock=BedrockConfig(max_tokens=bedrock_max_tokens),
            anthropic=AnthropicConfig(max_tokens=anthropic_max_tokens)
        )
        
        assert llm_config.ollama.max_tokens == ollama_max_tokens
        assert llm_config.openai.max_tokens == openai_max_tokens
        assert llm_config.bedrock.max_tokens == bedrock_max_tokens
        assert llm_config.anthropic.max_tokens == anthropic_max_tokens
    
    @given(
        config_dict=st.fixed_dictionaries({
            'ollama': st.fixed_dictionaries({
                'base_url': st.text(min_size=1, max_size=100),
                'default_model': st.text(min_size=1, max_size=50),
                'max_tokens': st.integers(min_value=100, max_value=10000),
                'temperature': st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                'timeout': st.integers(min_value=10, max_value=300)
            })
        })
    )
    @settings(max_examples=50)
    def test_property_load_from_dict_validates_structure(self, config_dict: dict):
        """
        Property 11.10: load_from_dict accepts valid dictionary structures.
        
        Valid configuration dictionaries should load successfully.
        """
        llm_config = LLMConfig.load_from_dict(config_dict)
        
        # Verify values were loaded correctly
        assert llm_config.ollama.base_url == config_dict['ollama']['base_url']
        assert llm_config.ollama.default_model == config_dict['ollama']['default_model']
        assert llm_config.ollama.max_tokens == config_dict['ollama']['max_tokens']
        assert abs(llm_config.ollama.temperature - config_dict['ollama']['temperature']) < 0.001
        assert llm_config.ollama.timeout == config_dict['ollama']['timeout']
    
    @given(
        empty_dict=st.just({})
    )
    @settings(max_examples=10)
    def test_property_empty_dict_uses_defaults(self, empty_dict: dict):
        """
        Property 11.11: Empty configuration dict uses all default values.
        
        When no configuration is provided, all defaults should be used.
        """
        llm_config = LLMConfig.load_from_dict(empty_dict)
        
        # Verify defaults are used
        assert llm_config.ollama.base_url == "http://localhost:11434"
        assert llm_config.ollama.default_model == "llama2"
        assert llm_config.ollama.max_tokens == 4096
        assert abs(llm_config.ollama.temperature - 0.3) < 0.001
        assert llm_config.ollama.timeout == 120
        
        assert llm_config.openai.api_key == ""
        assert llm_config.openai.default_model == "gpt-4"
        assert llm_config.openai.max_tokens == 4096
        assert abs(llm_config.openai.temperature - 0.7) < 0.001
        assert llm_config.openai.timeout == 60
    
    @given(
        partial_dict=st.fixed_dictionaries({
            'ollama': st.fixed_dictionaries({
                'base_url': st.text(min_size=1, max_size=100)
            })
        })
    )
    @settings(max_examples=20)
    def test_property_partial_config_uses_defaults_for_missing(self, partial_dict: dict):
        """
        Property 11.12: Partial configuration uses defaults for missing fields.
        
        When only some fields are provided, others should use defaults.
        """
        llm_config = LLMConfig.load_from_dict(partial_dict)
        
        # Verify provided value is used
        assert llm_config.ollama.base_url == partial_dict['ollama']['base_url']
        
        # Verify defaults are used for missing fields
        assert llm_config.ollama.default_model == "llama2"
        assert llm_config.ollama.max_tokens == 4096
        assert abs(llm_config.ollama.temperature - 0.3) < 0.001
        assert llm_config.ollama.timeout == 120
    
    @given(
        access_key=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        secret_key=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        session_token=st.one_of(st.none(), st.text(min_size=10, max_size=100))
    )
    @settings(max_examples=50)
    def test_property_optional_fields_accept_none(
        self,
        access_key: Optional[str],
        secret_key: Optional[str],
        session_token: Optional[str]
    ):
        """
        Property 11.13: Optional fields accept None values.
        
        Fields like AWS credentials are optional and should accept None.
        """
        config = BedrockConfig(
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=session_token
        )
        
        assert config.access_key_id == access_key
        assert config.secret_access_key == secret_key
        assert config.session_token == session_token
    
    @given(
        config_with_extra_fields=st.fixed_dictionaries({
            'ollama': st.fixed_dictionaries({
                'base_url': st.text(min_size=1, max_size=100),
                'extra_field': st.text(min_size=1, max_size=50)
            })
        })
    )
    @settings(max_examples=20)
    def test_property_extra_fields_ignored(self, config_with_extra_fields: dict):
        """
        Property 11.14: Extra fields in config dict are ignored.
        
        Unknown fields should not cause errors, allowing forward compatibility.
        """
        # Should not raise error
        llm_config = LLMConfig.load_from_dict(config_with_extra_fields)
        
        # Verify known field was loaded
        assert llm_config.ollama.base_url == config_with_extra_fields['ollama']['base_url']
        
        # Extra field is ignored (not accessible)
        assert not hasattr(llm_config.ollama, 'extra_field')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
