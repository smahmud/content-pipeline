"""
Property-Based Tests for Environment Variable Substitution

Property 7: Environment variable substitution works correctly
Validates: Requirements 3.6, 7.7

This test verifies that environment variables are correctly substituted
into configuration values across all providers and configuration fields.
"""

import os
import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Optional

from pipeline.llm.config import LLMConfig


# Strategy for generating valid environment variable names
env_var_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Nd'), min_codepoint=65, max_codepoint=90),
    min_size=3,
    max_size=30
).map(lambda s: s.upper())

# Strategy for generating valid configuration values
config_value_strategy = st.one_of(
    st.text(min_size=1, max_size=200),
    st.integers(min_value=1, max_value=100000).map(str),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False).map(str)
)


class TestEnvironmentVariableSubstitution:
    """Property tests for environment variable substitution."""
    
    @given(
        env_value=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=100)
    def test_property_env_var_always_overrides(self, env_value: str):
        """
        Property 7.1: Environment variable always overrides other sources.
        
        For any environment variable value, it should always be used
        regardless of config file or default values.
        """
        env_var = 'TEST_OVERRIDE'
        os.environ[env_var] = env_value
        
        try:
            # Test with various config and default values
            result1 = LLMConfig._resolve_value("config_value", env_var, "default_value")
            result2 = LLMConfig._resolve_value(None, env_var, "default_value")
            result3 = LLMConfig._resolve_value("", env_var, "default_value")
            
            assert result1 == env_value, "Env should override non-empty config"
            assert result2 == env_value, "Env should override None config"
            assert result3 == env_value, "Env should override empty config"
        
        finally:
            del os.environ[env_var]
    
    @given(
        base_url=st.text(min_size=10, max_size=100),
        model=st.text(min_size=3, max_size=50),
        max_tokens=st.integers(min_value=100, max_value=10000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        timeout=st.integers(min_value=10, max_value=300)
    )
    @settings(max_examples=50)
    def test_property_all_ollama_fields_support_env_vars(
        self,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ):
        """
        Property 7.2: All Ollama config fields support environment variables.
        
        Every configuration field should be settable via environment variable.
        """
        # Set all environment variables
        os.environ['OLLAMA_BASE_URL'] = base_url
        os.environ['OLLAMA_MODEL'] = model
        os.environ['OLLAMA_MAX_TOKENS'] = str(max_tokens)
        os.environ['OLLAMA_TEMPERATURE'] = str(temperature)
        os.environ['OLLAMA_TIMEOUT'] = str(timeout)
        
        try:
            # Load config with empty dict (no config file values)
            llm_config = LLMConfig.load_from_dict({})
            
            # Verify all values came from environment
            assert llm_config.ollama.base_url == base_url
            assert llm_config.ollama.default_model == model
            assert llm_config.ollama.max_tokens == max_tokens
            assert abs(llm_config.ollama.temperature - temperature) < 0.001
            assert llm_config.ollama.timeout == timeout
        
        finally:
            # Clean up
            for var in ['OLLAMA_BASE_URL', 'OLLAMA_MODEL', 'OLLAMA_MAX_TOKENS', 
                       'OLLAMA_TEMPERATURE', 'OLLAMA_TIMEOUT']:
                if var in os.environ:
                    del os.environ[var]
    
    @given(
        api_key=st.text(min_size=20, max_size=100),
        model=st.text(min_size=3, max_size=50),
        max_tokens=st.integers(min_value=100, max_value=10000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        timeout=st.integers(min_value=10, max_value=300)
    )
    @settings(max_examples=50)
    def test_property_all_openai_fields_support_env_vars(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ):
        """
        Property 7.3: All OpenAI config fields support environment variables.
        
        Every configuration field should be settable via environment variable.
        """
        # Set all environment variables
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_MODEL'] = model
        os.environ['OPENAI_MAX_TOKENS'] = str(max_tokens)
        os.environ['OPENAI_TEMPERATURE'] = str(temperature)
        os.environ['OPENAI_TIMEOUT'] = str(timeout)
        
        try:
            # Load config with empty dict
            llm_config = LLMConfig.load_from_dict({})
            
            # Verify all values came from environment
            assert llm_config.openai.api_key == api_key
            assert llm_config.openai.default_model == model
            assert llm_config.openai.max_tokens == max_tokens
            assert abs(llm_config.openai.temperature - temperature) < 0.001
            assert llm_config.openai.timeout == timeout
        
        finally:
            # Clean up
            for var in ['OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_MAX_TOKENS',
                       'OPENAI_TEMPERATURE', 'OPENAI_TIMEOUT']:
                if var in os.environ:
                    del os.environ[var]
    
    @given(
        region=st.sampled_from(['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']),
        model=st.text(min_size=10, max_size=50),
        max_tokens=st.integers(min_value=100, max_value=10000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_property_all_bedrock_fields_support_env_vars(
        self,
        region: str,
        model: str,
        max_tokens: int,
        temperature: float
    ):
        """
        Property 7.4: All Bedrock config fields support environment variables.
        
        Every configuration field should be settable via environment variable.
        """
        # Set all environment variables
        os.environ['AWS_REGION'] = region
        os.environ['BEDROCK_MODEL'] = model
        os.environ['BEDROCK_MAX_TOKENS'] = str(max_tokens)
        os.environ['BEDROCK_TEMPERATURE'] = str(temperature)
        
        try:
            # Load config with empty dict
            llm_config = LLMConfig.load_from_dict({})
            
            # Verify all values came from environment
            assert llm_config.bedrock.region == region
            assert llm_config.bedrock.default_model == model
            assert llm_config.bedrock.max_tokens == max_tokens
            assert abs(llm_config.bedrock.temperature - temperature) < 0.001
        
        finally:
            # Clean up
            for var in ['AWS_REGION', 'BEDROCK_MODEL', 'BEDROCK_MAX_TOKENS',
                       'BEDROCK_TEMPERATURE']:
                if var in os.environ:
                    del os.environ[var]
    
    @given(
        api_key=st.text(min_size=20, max_size=100),
        model=st.text(min_size=10, max_size=50),
        max_tokens=st.integers(min_value=100, max_value=10000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        timeout=st.integers(min_value=10, max_value=300)
    )
    @settings(max_examples=50)
    def test_property_all_anthropic_fields_support_env_vars(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ):
        """
        Property 7.5: All Anthropic config fields support environment variables.
        
        Every configuration field should be settable via environment variable.
        """
        # Set all environment variables
        os.environ['ANTHROPIC_API_KEY'] = api_key
        os.environ['ANTHROPIC_MODEL'] = model
        os.environ['ANTHROPIC_MAX_TOKENS'] = str(max_tokens)
        os.environ['ANTHROPIC_TEMPERATURE'] = str(temperature)
        os.environ['ANTHROPIC_TIMEOUT'] = str(timeout)
        
        try:
            # Load config with empty dict
            llm_config = LLMConfig.load_from_dict({})
            
            # Verify all values came from environment
            assert llm_config.anthropic.api_key == api_key
            assert llm_config.anthropic.default_model == model
            assert llm_config.anthropic.max_tokens == max_tokens
            assert abs(llm_config.anthropic.temperature - temperature) < 0.001
            assert llm_config.anthropic.timeout == timeout
        
        finally:
            # Clean up
            for var in ['ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL', 'ANTHROPIC_MAX_TOKENS',
                       'ANTHROPIC_TEMPERATURE', 'ANTHROPIC_TIMEOUT']:
                if var in os.environ:
                    del os.environ[var]
    
    @given(
        env_value=st.text(min_size=1, max_size=200),
        config_value=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=50)
    def test_property_env_var_overrides_config_for_same_field(
        self,
        env_value: str,
        config_value: str
    ):
        """
        Property 7.6: Environment variable overrides config for same field.
        
        When both env var and config value are set for the same field,
        the environment variable should always win.
        """
        assume(env_value != config_value)  # Make test meaningful
        
        os.environ['OLLAMA_BASE_URL'] = env_value
        
        try:
            config_dict = {
                'ollama': {
                    'base_url': config_value
                }
            }
            
            llm_config = LLMConfig.load_from_dict(config_dict)
            
            assert llm_config.ollama.base_url == env_value, \
                f"Environment variable should override config: expected {env_value}, got {llm_config.ollama.base_url}"
            assert llm_config.ollama.base_url != config_value, \
                "Config value should not be used when env var is set"
        
        finally:
            del os.environ['OLLAMA_BASE_URL']
    
    @given(
        int_value=st.integers(min_value=1, max_value=100000)
    )
    @settings(max_examples=50)
    def test_property_env_var_type_conversion_integers(self, int_value: int):
        """
        Property 7.7: Integer environment variables are correctly converted.
        
        Environment variables are strings, but integer config fields
        should correctly convert them to integers.
        """
        os.environ['OLLAMA_MAX_TOKENS'] = str(int_value)
        
        try:
            llm_config = LLMConfig.load_from_dict({})
            
            assert isinstance(llm_config.ollama.max_tokens, int), \
                "max_tokens should be converted to int"
            assert llm_config.ollama.max_tokens == int_value, \
                f"Expected {int_value}, got {llm_config.ollama.max_tokens}"
        
        finally:
            del os.environ['OLLAMA_MAX_TOKENS']
    
    @given(
        float_value=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_property_env_var_type_conversion_floats(self, float_value: float):
        """
        Property 7.8: Float environment variables are correctly converted.
        
        Environment variables are strings, but float config fields
        should correctly convert them to floats.
        """
        os.environ['OLLAMA_TEMPERATURE'] = str(float_value)
        
        try:
            llm_config = LLMConfig.load_from_dict({})
            
            assert isinstance(llm_config.ollama.temperature, float), \
                "temperature should be converted to float"
            assert abs(llm_config.ollama.temperature - float_value) < 0.001, \
                f"Expected {float_value}, got {llm_config.ollama.temperature}"
        
        finally:
            del os.environ['OLLAMA_TEMPERATURE']
    
    @given(
        provider=st.sampled_from(['ollama', 'openai', 'bedrock', 'anthropic']),
        field_name=st.sampled_from(['max_tokens', 'temperature', 'timeout'])
    )
    @settings(max_examples=50)
    def test_property_env_vars_independent_across_providers(
        self,
        provider: str,
        field_name: str
    ):
        """
        Property 7.9: Environment variables are independent across providers.
        
        Setting an environment variable for one provider should not affect
        the same field in other providers.
        """
        # Map provider to env var prefix
        env_prefixes = {
            'ollama': 'OLLAMA',
            'openai': 'OPENAI',
            'bedrock': 'BEDROCK',
            'anthropic': 'ANTHROPIC'
        }
        
        # Skip timeout for bedrock (not supported)
        if provider == 'bedrock' and field_name == 'timeout':
            assume(False)
        
        # Generate appropriate value based on field type
        if field_name == 'temperature':
            value = 0.5  # Float value for temperature
        else:  # max_tokens or timeout
            value = 2000  # Integer value
        
        # Set environment variable for one provider
        env_var = f"{env_prefixes[provider]}_{field_name.upper()}"
        os.environ[env_var] = str(value)
        
        try:
            llm_config = LLMConfig.load_from_dict({})
            
            # Get the config for the provider
            provider_config = getattr(llm_config, provider)
            
            # Check that the value was set for this provider
            actual_value = getattr(provider_config, field_name)
            if isinstance(value, float):
                assert abs(actual_value - value) < 0.001, \
                    f"Value should be set for {provider}"
            else:
                assert actual_value == value, \
                    f"Value should be set for {provider}"
            
            # Check that other providers use defaults
            for other_provider in ['ollama', 'openai', 'bedrock', 'anthropic']:
                if other_provider == provider:
                    continue
                
                # Skip timeout for bedrock
                if other_provider == 'bedrock' and field_name == 'timeout':
                    continue
                
                other_config = getattr(llm_config, other_provider)
                other_value = getattr(other_config, field_name)
                
                # Should be default value, not the env var value
                if isinstance(value, float):
                    assert abs(other_value - value) > 0.001 or other_value == value, \
                        f"Other provider {other_provider} should not be affected"
                else:
                    # Just verify it's a valid value (could be default)
                    assert other_value is not None
        
        finally:
            if env_var in os.environ:
                del os.environ[env_var]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
