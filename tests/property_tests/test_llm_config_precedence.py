"""
Property-Based Tests for LLM Configuration Precedence

Property 6: Configuration precedence is respected
Validates: Requirements 3.7, 7.5, 7.6

This test verifies that configuration values are resolved with the correct precedence:
1. Environment variables (highest priority)
2. Config file values (medium priority)
3. Default values (lowest priority)

The property holds for all possible combinations of configuration sources.
"""

import os
import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Optional, Any

from pipeline.llm.config import LLMConfig, OllamaConfig, OpenAIConfig, BedrockConfig, AnthropicConfig


# Strategy for generating valid configuration values
config_value_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=100),
    st.integers(min_value=1, max_value=10000),
    st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
)


class TestConfigurationPrecedence:
    """Property tests for configuration precedence rules."""
    
    @given(
        env_value=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        config_value=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        default_value=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_property_env_overrides_config_and_default(
        self,
        env_value: Optional[str],
        config_value: Optional[str],
        default_value: str
    ):
        """
        Property 6.1: Environment variable overrides config and default values.
        
        For any combination of env, config, and default values:
        - If env is set, result should be env value
        - Otherwise, if config is set, result should be config value
        - Otherwise, result should be default value
        """
        result = LLMConfig._resolve_value(config_value, 'TEST_ENV_VAR', default_value)
        
        if env_value is not None:
            # Set environment variable
            os.environ['TEST_ENV_VAR'] = env_value
            try:
                result = LLMConfig._resolve_value(config_value, 'TEST_ENV_VAR', default_value)
                assert result == env_value, \
                    f"Environment variable should override: expected {env_value}, got {result}"
            finally:
                # Clean up
                del os.environ['TEST_ENV_VAR']
        elif config_value is not None:
            assert result == config_value, \
                f"Config value should be used when env not set: expected {config_value}, got {result}"
        else:
            assert result == default_value, \
                f"Default value should be used when neither env nor config set: expected {default_value}, got {result}"
    
    @given(
        config_value=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        default_value=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_property_config_overrides_default(
        self,
        config_value: Optional[str],
        default_value: str
    ):
        """
        Property 6.2: Config value overrides default when env not set.
        
        When environment variable is not set:
        - If config is set, result should be config value
        - Otherwise, result should be default value
        """
        # Ensure env var is not set
        env_var = 'TEST_CONFIG_OVERRIDE'
        if env_var in os.environ:
            del os.environ[env_var]
        
        result = LLMConfig._resolve_value(config_value, env_var, default_value)
        
        if config_value is not None:
            assert result == config_value, \
                f"Config value should override default: expected {config_value}, got {result}"
        else:
            assert result == default_value, \
                f"Default value should be used when config not set: expected {default_value}, got {result}"
    
    @given(
        default_value=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_property_default_used_when_nothing_set(self, default_value: str):
        """
        Property 6.3: Default value used when neither env nor config set.
        
        When both environment variable and config value are None,
        the default value should always be returned.
        """
        env_var = 'TEST_DEFAULT_FALLBACK'
        if env_var in os.environ:
            del os.environ[env_var]
        
        result = LLMConfig._resolve_value(None, env_var, default_value)
        
        assert result == default_value, \
            f"Default value should be used: expected {default_value}, got {result}"
    
    @given(
        base_url_env=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        base_url_config=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        model_env=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        model_config=st.one_of(st.none(), st.text(min_size=1, max_size=50))
    )
    @settings(max_examples=50)
    def test_property_precedence_applies_to_all_config_fields(
        self,
        base_url_env: Optional[str],
        base_url_config: Optional[str],
        model_env: Optional[str],
        model_config: Optional[str]
    ):
        """
        Property 6.4: Precedence rules apply independently to each config field.
        
        Each configuration field should follow precedence rules independently.
        Setting env for one field should not affect precedence for other fields.
        """
        # Set up environment variables
        env_vars_to_clean = []
        
        if base_url_env is not None:
            os.environ['OLLAMA_BASE_URL'] = base_url_env
            env_vars_to_clean.append('OLLAMA_BASE_URL')
        
        if model_env is not None:
            os.environ['OLLAMA_MODEL'] = model_env
            env_vars_to_clean.append('OLLAMA_MODEL')
        
        try:
            # Create config dict
            config_dict = {
                'ollama': {}
            }
            if base_url_config is not None:
                config_dict['ollama']['base_url'] = base_url_config
            if model_config is not None:
                config_dict['ollama']['default_model'] = model_config
            
            # Load config
            llm_config = LLMConfig.load_from_dict(config_dict)
            
            # Verify base_url precedence
            if base_url_env is not None:
                assert llm_config.ollama.base_url == base_url_env, \
                    "base_url should use env value"
            elif base_url_config is not None:
                assert llm_config.ollama.base_url == base_url_config, \
                    "base_url should use config value"
            else:
                assert llm_config.ollama.base_url == "http://localhost:11434", \
                    "base_url should use default value"
            
            # Verify model precedence (independent of base_url)
            if model_env is not None:
                assert llm_config.ollama.default_model == model_env, \
                    "model should use env value"
            elif model_config is not None:
                assert llm_config.ollama.default_model == model_config, \
                    "model should use config value"
            else:
                assert llm_config.ollama.default_model == "llama2", \
                    "model should use default value"
        
        finally:
            # Clean up environment variables
            for env_var in env_vars_to_clean:
                if env_var in os.environ:
                    del os.environ[env_var]
    
    @given(
        api_key_env=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        api_key_config=st.one_of(st.none(), st.text(min_size=10, max_size=100))
    )
    @settings(max_examples=50)
    def test_property_precedence_for_sensitive_values(
        self,
        api_key_env: Optional[str],
        api_key_config: Optional[str]
    ):
        """
        Property 6.5: Precedence rules apply to sensitive values (API keys).
        
        API keys and other sensitive values should follow the same precedence
        rules as other configuration values.
        """
        env_var = 'OPENAI_API_KEY'
        
        # Set up environment
        if api_key_env is not None:
            os.environ[env_var] = api_key_env
        elif env_var in os.environ:
            del os.environ[env_var]
        
        try:
            # Create config dict
            config_dict = {
                'openai': {}
            }
            if api_key_config is not None:
                config_dict['openai']['api_key'] = api_key_config
            
            # Load config
            llm_config = LLMConfig.load_from_dict(config_dict)
            
            # Verify precedence
            if api_key_env is not None:
                assert llm_config.openai.api_key == api_key_env, \
                    "API key should use env value"
            elif api_key_config is not None:
                assert llm_config.openai.api_key == api_key_config, \
                    "API key should use config value"
            else:
                assert llm_config.openai.api_key == "", \
                    "API key should use default (empty string)"
        
        finally:
            # Clean up
            if env_var in os.environ:
                del os.environ[env_var]
    
    @given(
        max_tokens_env=st.one_of(st.none(), st.integers(min_value=100, max_value=10000).map(str)),
        max_tokens_config=st.one_of(st.none(), st.integers(min_value=100, max_value=10000)),
        temperature_env=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0).map(str)),
        temperature_config=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0))
    )
    @settings(max_examples=50)
    def test_property_precedence_with_type_conversion(
        self,
        max_tokens_env: Optional[str],
        max_tokens_config: Optional[int],
        temperature_env: Optional[str],
        temperature_config: Optional[float]
    ):
        """
        Property 6.6: Precedence rules work with type conversion.
        
        Environment variables are strings but config values may be typed.
        Precedence should work correctly even when type conversion is needed.
        """
        # Filter out invalid float strings
        if temperature_env is not None:
            try:
                float(temperature_env)
            except (ValueError, OverflowError):
                assume(False)
        
        # Set up environment
        env_vars_to_clean = []
        
        if max_tokens_env is not None:
            os.environ['OPENAI_MAX_TOKENS'] = max_tokens_env
            env_vars_to_clean.append('OPENAI_MAX_TOKENS')
        
        if temperature_env is not None:
            os.environ['OPENAI_TEMPERATURE'] = temperature_env
            env_vars_to_clean.append('OPENAI_TEMPERATURE')
        
        try:
            # Create config dict
            config_dict = {
                'openai': {}
            }
            if max_tokens_config is not None:
                config_dict['openai']['max_tokens'] = max_tokens_config
            if temperature_config is not None:
                config_dict['openai']['temperature'] = temperature_config
            
            # Load config
            llm_config = LLMConfig.load_from_dict(config_dict)
            
            # Verify max_tokens precedence with type conversion
            if max_tokens_env is not None:
                assert llm_config.openai.max_tokens == int(max_tokens_env), \
                    "max_tokens should use env value (converted to int)"
            elif max_tokens_config is not None:
                assert llm_config.openai.max_tokens == max_tokens_config, \
                    "max_tokens should use config value"
            else:
                assert llm_config.openai.max_tokens == 4096, \
                    "max_tokens should use default value"
            
            # Verify temperature precedence with type conversion
            if temperature_env is not None:
                assert abs(llm_config.openai.temperature - float(temperature_env)) < 0.001, \
                    "temperature should use env value (converted to float)"
            elif temperature_config is not None:
                assert abs(llm_config.openai.temperature - temperature_config) < 0.001, \
                    "temperature should use config value"
            else:
                assert abs(llm_config.openai.temperature - 0.7) < 0.001, \
                    "temperature should use default value"
        
        finally:
            # Clean up
            for env_var in env_vars_to_clean:
                if env_var in os.environ:
                    del os.environ[env_var]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
