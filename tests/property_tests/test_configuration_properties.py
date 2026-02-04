"""
Property-based tests for configuration schema validation.

**Property 7: Configuration Schema Validation**
*For any* configuration object, the Configuration_Manager should validate it against 
the schema and report specific errors for invalid configurations.
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""

import os
import pytest
from hypothesis import given, strategies as st, assume
from pipeline.config.schema import (
    TranscriptionConfig, 
    WhisperLocalConfig, 
    WhisperAPIConfig, 
    AWSTranscribeConfig,
    AutoSelectionConfig,
    EngineType,
    LogLevel,
    WhisperModelSize
)


# Strategy for generating valid engine types
valid_engines = st.sampled_from([e.value for e in EngineType])

# Strategy for generating valid log levels  
valid_log_levels = st.sampled_from([l.value for l in LogLevel])

# Strategy for generating valid Whisper model sizes
valid_whisper_models = st.sampled_from([m.value for m in WhisperModelSize])

# Strategy for generating positive integers (smaller range for faster testing)
positive_ints = st.integers(min_value=1, max_value=600)

# Strategy for generating non-negative integers (smaller range)
non_negative_ints = st.integers(min_value=0, max_value=5)

# Strategy for generating valid path-like text (ASCII printable, no special chars)
valid_path_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='\\/:*?"<>|'),
    min_size=1, 
    max_size=30
).filter(lambda x: x.strip() and not x.startswith('.') and not x.endswith('.'))


@given(
    engine=valid_engines,
    log_level=valid_log_levels,
    whisper_model=valid_whisper_models,
    timeout=positive_ints,
    retry_attempts=non_negative_ints
)
def test_valid_configuration_has_no_errors(engine, log_level, whisper_model, timeout, retry_attempts):
    """
    **Property 7: Configuration Schema Validation**
    *For any* valid configuration values, validation should return no errors.
    """
    config = TranscriptionConfig(
        engine=engine,
        log_level=log_level,
        whisper_local=WhisperLocalConfig(
            model=whisper_model,
            timeout=timeout,
            retry_attempts=retry_attempts
        ),
        whisper_api=WhisperAPIConfig(
            timeout=timeout,
            retry_attempts=retry_attempts
        ),
        aws_transcribe=AWSTranscribeConfig(
            timeout=timeout,
            retry_attempts=retry_attempts
        )
    )
    
    errors = config.validate()
    assert errors == [], f"Valid configuration should have no errors, got: {errors}"


@given(
    invalid_engine=st.text().filter(lambda x: x not in [e.value for e in EngineType]),
    valid_log_level=valid_log_levels,
    valid_whisper_model=valid_whisper_models
)
def test_invalid_engine_produces_error(invalid_engine, valid_log_level, valid_whisper_model):
    """
    **Property 7: Configuration Schema Validation**  
    *For any* invalid engine type, validation should return specific error about valid options.
    """
    assume(invalid_engine.strip() != "")  # Skip empty strings
    
    config = TranscriptionConfig(
        engine=invalid_engine,
        log_level=valid_log_level,
        whisper_local=WhisperLocalConfig(model=valid_whisper_model)
    )
    
    errors = config.validate()
    assert len(errors) > 0, "Invalid engine should produce validation errors"
    
    # Check that error mentions the invalid engine and valid options
    engine_error = next((e for e in errors if "Invalid engine" in e), None)
    assert engine_error is not None, f"Should have engine validation error, got: {errors}"
    assert invalid_engine in engine_error, f"Error should mention invalid engine '{invalid_engine}'"
    assert "Valid options:" in engine_error, "Error should list valid options"


@given(
    valid_engine=valid_engines,
    invalid_log_level=st.text().filter(lambda x: x not in [l.value for l in LogLevel]),
    valid_whisper_model=valid_whisper_models
)
def test_invalid_log_level_produces_error(valid_engine, invalid_log_level, valid_whisper_model):
    """
    **Property 7: Configuration Schema Validation**
    *For any* invalid log level, validation should return specific error about valid options.
    """
    assume(invalid_log_level.strip() != "")  # Skip empty strings
    
    config = TranscriptionConfig(
        engine=valid_engine,
        log_level=invalid_log_level,
        whisper_local=WhisperLocalConfig(model=valid_whisper_model)
    )
    
    errors = config.validate()
    assert len(errors) > 0, "Invalid log level should produce validation errors"
    
    # Check that error mentions the invalid log level and valid options
    log_error = next((e for e in errors if "Invalid log_level" in e), None)
    assert log_error is not None, f"Should have log level validation error, got: {errors}"
    assert invalid_log_level in log_error, f"Error should mention invalid log level '{invalid_log_level}'"
    assert "Valid options:" in log_error, "Error should list valid options"


@given(
    valid_engine=valid_engines,
    valid_log_level=valid_log_levels,
    invalid_whisper_model=st.text().filter(lambda x: x not in [m.value for m in WhisperModelSize])
)
def test_invalid_whisper_model_produces_error(valid_engine, valid_log_level, invalid_whisper_model):
    """
    **Property 7: Configuration Schema Validation**
    *For any* invalid Whisper model size, validation should return specific error about valid options.
    """
    assume(invalid_whisper_model.strip() != "")  # Skip empty strings
    
    config = TranscriptionConfig(
        engine=valid_engine,
        log_level=valid_log_level,
        whisper_local=WhisperLocalConfig(model=invalid_whisper_model)
    )
    
    errors = config.validate()
    assert len(errors) > 0, "Invalid Whisper model should produce validation errors"
    
    # Check that error mentions the invalid model and valid options
    model_error = next((e for e in errors if "Invalid whisper_local.model" in e), None)
    assert model_error is not None, f"Should have Whisper model validation error, got: {errors}"
    assert invalid_whisper_model in model_error, f"Error should mention invalid model '{invalid_whisper_model}'"
    assert "Valid options:" in model_error, "Error should list valid options"


@given(
    timeout=st.integers(max_value=0),  # Non-positive timeouts
    retry_attempts=st.integers(max_value=-1)  # Negative retry attempts
)
def test_invalid_numeric_values_produce_errors(timeout, retry_attempts):
    """
    **Property 7: Configuration Schema Validation**
    *For any* invalid numeric configuration values, validation should return specific errors.
    """
    config = TranscriptionConfig(
        whisper_local=WhisperLocalConfig(
            timeout=timeout,
            retry_attempts=retry_attempts
        ),
        whisper_api=WhisperAPIConfig(
            timeout=timeout,
            retry_attempts=retry_attempts
        ),
        aws_transcribe=AWSTranscribeConfig(
            timeout=timeout,
            retry_attempts=retry_attempts
        )
    )
    
    errors = config.validate()
    assert len(errors) > 0, "Invalid numeric values should produce validation errors"
    
    if timeout <= 0:
        timeout_errors = [e for e in errors if "timeout must be positive" in e]
        assert len(timeout_errors) >= 1, f"Should have timeout validation errors, got: {errors}"
    
    if retry_attempts < 0:
        retry_errors = [e for e in errors if "retry_attempts must be non-negative" in e]
        assert len(retry_errors) >= 1, f"Should have retry_attempts validation errors, got: {errors}"


def test_get_engine_config_returns_correct_config():
    """
    **Property 7: Configuration Schema Validation**
    *For any* valid engine type, get_engine_config should return the corresponding configuration.
    """
    config = TranscriptionConfig()
    
    # Test each engine type
    local_config = config.get_engine_config(EngineType.WHISPER_LOCAL.value)
    assert isinstance(local_config, WhisperLocalConfig)
    
    api_config = config.get_engine_config(EngineType.WHISPER_API.value)
    assert isinstance(api_config, WhisperAPIConfig)
    
    aws_config = config.get_engine_config(EngineType.AWS_TRANSCRIBE.value)
    assert isinstance(aws_config, AWSTranscribeConfig)
    
    # Test invalid engine type
    with pytest.raises(ValueError, match="Unknown engine type"):
        config.get_engine_config("invalid-engine")


@given(st.data())
def test_configuration_validation_is_comprehensive(data):
    """
    **Property 7: Configuration Schema Validation**
    *For any* configuration with multiple invalid values, validation should catch all errors.
    """
    # Generate a configuration with potentially multiple invalid values
    config = TranscriptionConfig(
        engine=data.draw(st.text()),
        log_level=data.draw(st.text()),
        whisper_local=WhisperLocalConfig(
            model=data.draw(st.text()),
            timeout=data.draw(st.integers(max_value=0)),
            retry_attempts=data.draw(st.integers(max_value=-1))
        )
    )
    
    errors = config.validate()
    
    # If any values are invalid, there should be errors
    valid_engines = [e.value for e in EngineType]
    valid_log_levels = [l.value for l in LogLevel]
    valid_models = [m.value for m in WhisperModelSize]
    
    has_invalid_values = (
        config.engine not in valid_engines or
        config.log_level not in valid_log_levels or
        config.whisper_local.model not in valid_models or
        config.whisper_local.timeout <= 0 or
        config.whisper_local.retry_attempts < 0
    )
    
    if has_invalid_values:
        assert len(errors) > 0, f"Configuration with invalid values should have errors: {config}"
    
    # Each error should be specific and actionable
    for error in errors:
        assert len(error.strip()) > 0, "Errors should not be empty"
        assert any(keyword in error.lower() for keyword in ["invalid", "must be", "valid options"]), \
            f"Error should be descriptive and actionable: '{error}'"


"""
Property-based tests for configuration precedence.

**Property 2: Configuration Precedence**
*For any* combination of configuration sources, the final configuration should follow 
the precedence order: CLI flags > environment variables > configuration files > defaults.
**Validates: Requirements 5.4, 6.1, 6.2, 6.3**
"""

import tempfile
from pathlib import Path
from unittest.mock import patch
from pipeline.config.manager import ConfigurationManager


def test_cli_overrides_have_highest_precedence():
    """
    **Property 2: Configuration Precedence**
    CLI overrides should always take precedence over all other configuration sources.
    """
    config_manager = ConfigurationManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config file with specific values
        config_content = """
engine: local-whisper
output_dir: ./config-output
log_level: debug
"""
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(config_content)
        
        # Set environment variables
        env_vars = {
            'CONTENT_PIPELINE_DEFAULT_ENGINE': 'aws-transcribe',
            'CONTENT_PIPELINE_OUTPUT_DIR': './env-output'
        }
        
        # CLI overrides
        cli_overrides = {
            'engine': 'openai-whisper',
            'output_dir': './cli-output',
            'log_level': 'error'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch.object(config_manager, 'user_config_path', config_path):
            
            config = config_manager.load_configuration(cli_overrides=cli_overrides)
            
            # CLI overrides should always win
            assert config.engine == 'openai-whisper'
            assert config.output_dir == './cli-output'
            assert config.log_level == 'error'


def test_environment_variables_override_config_files():
    """
    **Property 2: Configuration Precedence**
    Environment variables should override configuration files but not CLI arguments.
    """
    config_manager = ConfigurationManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config file
        config_content = """
engine: local-whisper
output_dir: ./config-output
"""
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(config_content)
        
        # Set environment variables
        env_vars = {
            'CONTENT_PIPELINE_DEFAULT_ENGINE': 'aws-transcribe',
            'CONTENT_PIPELINE_OUTPUT_DIR': './env-output'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch.object(config_manager, 'user_config_path', config_path):
            
            config = config_manager.load_configuration()
            
            # Environment variables should override config file
            assert config.engine == 'aws-transcribe'
            assert config.output_dir == './env-output'


def test_project_config_overrides_user_config():
    """
    **Property 2: Configuration Precedence**
    Project configuration should override user configuration.
    """
    config_manager = ConfigurationManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create user config
        user_config_content = """
engine: local-whisper
output_dir: ./user-output
log_level: debug
"""
        user_config_path = Path(temp_dir) / "user.yaml"
        user_config_path.write_text(user_config_content)
        
        # Create project config (overrides some values)
        project_config_content = """
engine: openai-whisper
log_level: warning
"""
        project_config_path = Path(temp_dir) / "project.yaml"
        project_config_path.write_text(project_config_content)
        
        with patch.object(config_manager, 'user_config_path', user_config_path), \
             patch.object(config_manager, 'project_config_path', project_config_path):
            
            config = config_manager.load_configuration()
            
            # Project config should override user config
            assert config.engine == 'openai-whisper'
            assert config.log_level == 'warning'
            
            # Non-overridden values should be preserved from user config
            assert config.output_dir == './user-output'


def test_nested_configuration_merging():
    """
    **Property 2: Configuration Precedence**
    Nested configuration structures should merge properly, preserving non-overridden values.
    """
    config_manager = ConfigurationManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create base config with nested structure
        base_config_content = """
engine: local-whisper
whisper_local:
  model: base
  device: cpu
  timeout: 300
"""
        base_config_path = Path(temp_dir) / "base.yaml"
        base_config_path.write_text(base_config_content)
        
        # Create override config (partial nested override)
        override_config_content = """
engine: openai-whisper
whisper_local:
  model: large
  timeout: 600
"""
        override_config_path = Path(temp_dir) / "override.yaml"
        override_config_path.write_text(override_config_content)
        
        with patch.object(config_manager, 'user_config_path', base_config_path), \
             patch.object(config_manager, 'project_config_path', override_config_path):
            
            config = config_manager.load_configuration()
            
            # Override values should win
            assert config.engine == 'openai-whisper'
            assert config.whisper_local.model == 'large'
            assert config.whisper_local.timeout == 600
            
            # Non-overridden nested values should be preserved
            assert config.whisper_local.device == 'cpu'


def test_full_precedence_chain():
    """
    **Property 2: Configuration Precedence**
    Complete precedence chain: CLI > Environment > Project > User > Defaults
    """
    config_manager = ConfigurationManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create user config
        user_config_content = "engine: local-whisper"
        user_config_path = Path(temp_dir) / "user.yaml"
        user_config_path.write_text(user_config_content)
        
        # Create project config
        project_config_content = "engine: aws-transcribe"
        project_config_path = Path(temp_dir) / "project.yaml"
        project_config_path.write_text(project_config_content)
        
        with patch.object(config_manager, 'user_config_path', user_config_path), \
             patch.object(config_manager, 'project_config_path', project_config_path):
            
            # Test 1: Only configs (project should override user)
            config = config_manager.load_configuration()
            assert config.engine == 'aws-transcribe'
            
            # Test 2: Add environment variable (should override project)
            with patch.dict(os.environ, {'CONTENT_PIPELINE_DEFAULT_ENGINE': 'openai-whisper'}):
                config = config_manager.load_configuration()
                assert config.engine == 'openai-whisper'
                
                # Test 3: Add CLI override (should override environment)
                cli_overrides = {'engine': 'local-whisper'}
                config = config_manager.load_configuration(cli_overrides=cli_overrides)
                assert config.engine == 'local-whisper'


"""
Property-based tests for environment variable substitution.

**Property 3: Environment Variable Substitution**
*For any* configuration file containing ${VARIABLE} syntax, the Configuration_Manager 
should substitute environment variable values correctly and provide clear errors for missing variables.
**Validates: Requirements 6.6, 11.1, 11.2**
"""

import string
from hypothesis import given, strategies as st, assume
from pipeline.config.manager import ConfigurationManager


# Strategy for generating valid environment variable names (alphanumeric + underscore)
valid_env_var_names = st.text(
    alphabet=string.ascii_uppercase + string.digits + '_',
    min_size=1,
    max_size=20
).filter(lambda x: x[0].isalpha() and not x.endswith('_'))

# Strategy for generating valid environment variable values (printable ASCII, no special chars)
valid_env_var_values = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='${}'),
    min_size=0,
    max_size=50
)

# Strategy for generating valid default values
valid_default_values = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='${}'),
    min_size=0,
    max_size=30
)


@given(
    var_name=valid_env_var_names,
    var_value=valid_env_var_values
)
def test_environment_variable_substitution_with_value(var_name, var_value):
    """
    **Property 3: Environment Variable Substitution**
    *For any* environment variable with a value, ${VAR} substitution should replace with the actual value.
    """
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'${{{var_name}}}'
    }
    
    with patch.dict(os.environ, {var_name: var_value}):
        result = config_manager.substitute_environment_variables(config_dict)
        assert result['test_field'] == var_value


@given(
    var_name=valid_env_var_names,
    var_value=valid_env_var_values,
    default_value=valid_default_values
)
def test_environment_variable_substitution_with_default_used(var_name, var_value, default_value):
    """
    **Property 3: Environment Variable Substitution**
    *For any* environment variable with a value, ${VAR:-default} should use the actual value, not the default.
    """
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'${{{var_name}:-{default_value}}}'
    }
    
    with patch.dict(os.environ, {var_name: var_value}):
        result = config_manager.substitute_environment_variables(config_dict)
        assert result['test_field'] == var_value


@given(
    var_name=valid_env_var_names,
    default_value=valid_default_values
)
def test_environment_variable_substitution_with_default_fallback(var_name, default_value):
    """
    **Property 3: Environment Variable Substitution**
    *For any* missing environment variable, ${VAR:-default} should use the default value.
    """
    assume(var_name not in os.environ)  # Ensure the variable is not set
    
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'${{{var_name}:-{default_value}}}'
    }
    
    # Clear the environment variable to ensure it's not set
    with patch.dict(os.environ, {}, clear=False):
        if var_name in os.environ:
            del os.environ[var_name]
        
        result = config_manager.substitute_environment_variables(config_dict)
        assert result['test_field'] == default_value


@given(var_name=valid_env_var_names)
def test_environment_variable_substitution_missing_required_raises_error(var_name):
    """
    **Property 3: Environment Variable Substitution**
    *For any* missing required environment variable, ${VAR} should raise a clear error.
    """
    assume(var_name not in os.environ)  # Ensure the variable is not set
    
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'${{{var_name}}}'
    }
    
    # Clear the environment variable to ensure it's not set
    with patch.dict(os.environ, {}, clear=False):
        if var_name in os.environ:
            del os.environ[var_name]
        
        with pytest.raises(ValueError, match=f"Required environment variable '{var_name}' is not set"):
            config_manager.substitute_environment_variables(config_dict)


@given(st.data())
def test_nested_environment_variable_substitution(data):
    """
    **Property 3: Environment Variable Substitution**
    *For any* nested configuration structure, environment variable substitution should work recursively.
    """
    config_manager = ConfigurationManager()
    
    # Generate test data
    var_name = data.draw(valid_env_var_names)
    var_value = data.draw(valid_env_var_values)
    
    config_dict = {
        'level1': {
            'level2': {
                'test_field': f'${{{var_name}}}'
            },
            'array': [
                f'${{{var_name}}}',
                'static_value'
            ]
        },
        'root_field': f'prefix-${{{var_name}}}-suffix'
    }
    
    with patch.dict(os.environ, {var_name: var_value}):
        result = config_manager.substitute_environment_variables(config_dict)
        
        assert result['level1']['level2']['test_field'] == var_value
        assert result['level1']['array'][0] == var_value
        assert result['level1']['array'][1] == 'static_value'
        assert result['root_field'] == f'prefix-{var_value}-suffix'


@given(
    var_name=valid_env_var_names,
    var_value=valid_env_var_values
)
def test_partial_environment_variable_substitution(var_name, var_value):
    """
    **Property 3: Environment Variable Substitution**
    *For any* string containing environment variables mixed with other text, 
    substitution should only replace the variable parts.
    """
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'prefix-${{{var_name}}}-suffix'
    }
    
    with patch.dict(os.environ, {var_name: var_value}):
        result = config_manager.substitute_environment_variables(config_dict)
        assert result['test_field'] == f'prefix-{var_value}-suffix'


def test_non_string_values_unchanged():
    """
    **Property 3: Environment Variable Substitution**
    *For any* non-string configuration values, substitution should leave them unchanged.
    """
    config_manager = ConfigurationManager()
    
    config_dict = {
        'integer': 42,
        'float': 3.14,
        'boolean': True,
        'null': None,
        'list': [1, 2, 3],
        'dict': {'nested': 'value'}
    }
    
    result = config_manager.substitute_environment_variables(config_dict)
    assert result == config_dict


@given(
    var1_name=valid_env_var_names,
    var1_value=valid_env_var_values,
    var2_name=valid_env_var_names,
    var2_value=valid_env_var_values
)
def test_multiple_environment_variables_in_single_string(var1_name, var1_value, var2_name, var2_value):
    """
    **Property 3: Environment Variable Substitution**
    *For any* string containing multiple environment variables, all should be substituted correctly.
    """
    assume(var1_name != var2_name)  # Ensure different variable names
    
    config_manager = ConfigurationManager()
    
    config_dict = {
        'test_field': f'${{{var1_name}}}-${{{var2_name}}}'
    }
    
    with patch.dict(os.environ, {var1_name: var1_value, var2_name: var2_value}):
        result = config_manager.substitute_environment_variables(config_dict)
        assert result['test_field'] == f'{var1_value}-{var2_value}'


"""
Property-based tests for YAML round-trip consistency.

**Property 8: YAML Round-Trip Consistency**
*For any* valid configuration object, parsing then printing then parsing should 
produce an equivalent object (round-trip property).
**Validates: Requirements 5.6, 5.7**
"""

from hypothesis import given, strategies as st
from pipeline.config.schema import TranscriptionConfig, EngineType, LogLevel, WhisperModelSize
from pipeline.config.yaml_parser import ConfigurationYAMLParser
from pipeline.config.pretty_printer import ConfigurationPrettyPrinter


def test_yaml_round_trip_with_default_config():
    """
    **Property 8: YAML Round-Trip Consistency**
    Default configuration should round-trip correctly through YAML serialization.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    # Start with default configuration
    config = TranscriptionConfig()
    
    # Serialize to YAML without comments (cleaner for round-trip)
    yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
    
    # Parse back to dictionary
    parsed_dict = yaml_parser.parse_string(yaml_content)
    
    # Convert back to configuration object
    from pipeline.config.manager import ConfigurationManager
    config_manager = ConfigurationManager()
    parsed_config = config_manager._dict_to_config(parsed_dict)
    
    # All essential values should be preserved
    assert parsed_config.engine == config.engine
    assert parsed_config.output_dir == config.output_dir
    assert parsed_config.log_level == config.log_level
    assert parsed_config.language == config.language
    
    # Nested configurations should be preserved
    assert parsed_config.whisper_local.model == config.whisper_local.model
    assert parsed_config.whisper_local.device == config.whisper_local.device
    assert parsed_config.whisper_local.timeout == config.whisper_local.timeout
    
    assert parsed_config.whisper_api.model == config.whisper_api.model
    assert parsed_config.whisper_api.temperature == config.whisper_api.temperature
    
    assert parsed_config.aws_transcribe.region == config.aws_transcribe.region
    assert parsed_config.aws_transcribe.language_code == config.aws_transcribe.language_code
    
    assert parsed_config.auto_selection.prefer_local == config.auto_selection.prefer_local
    assert parsed_config.auto_selection.fallback_enabled == config.auto_selection.fallback_enabled
    assert parsed_config.auto_selection.priority_order == config.auto_selection.priority_order


def test_yaml_round_trip_with_custom_config():
    """
    **Property 8: YAML Round-Trip Consistency**
    Custom configuration should round-trip correctly through YAML serialization.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    # Create custom configuration with safe values
    config = TranscriptionConfig(
        engine="local-whisper",
        output_dir="./custom-output",
        log_level="debug",
        language="en"
    )
    config.whisper_local.model = "large"
    config.whisper_local.timeout = 600
    config.whisper_api.temperature = 0.5
    config.aws_transcribe.region = "us-west-2"
    
    # Serialize to YAML
    yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
    
    # Parse back
    parsed_dict = yaml_parser.parse_string(yaml_content)
    
    # Convert back to configuration object
    from pipeline.config.manager import ConfigurationManager
    config_manager = ConfigurationManager()
    parsed_config = config_manager._dict_to_config(parsed_dict)
    
    # All values should be exactly preserved
    assert parsed_config.engine == "local-whisper"
    assert parsed_config.output_dir == "./custom-output"
    assert parsed_config.log_level == "debug"
    assert parsed_config.language == "en"
    
    assert parsed_config.whisper_local.model == "large"
    assert parsed_config.whisper_local.timeout == 600
    assert abs(parsed_config.whisper_api.temperature - 0.5) < 0.001
    assert parsed_config.aws_transcribe.region == "us-west-2"


def test_yaml_round_trip_with_comments():
    """
    **Property 8: YAML Round-Trip Consistency**
    Configuration with comments should preserve essential values after round-trip.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    config = TranscriptionConfig(
        engine="openai-whisper",
        output_dir="./api-output",
        log_level="warning"
    )
    
    # Serialize with comments
    yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=True)
    
    # Parse back (comments will be stripped)
    parsed_dict = yaml_parser.parse_string(yaml_content)
    
    # Convert back to configuration object
    from pipeline.config.manager import ConfigurationManager
    config_manager = ConfigurationManager()
    parsed_config = config_manager._dict_to_config(parsed_dict)
    
    # Essential values should be preserved
    assert parsed_config.engine == "openai-whisper"
    assert parsed_config.output_dir == "./api-output"
    assert parsed_config.log_level == "warning"


def test_pretty_printer_round_trip():
    """
    **Property 8: YAML Round-Trip Consistency**
    Pretty printer output should round-trip correctly.
    """
    pretty_printer = ConfigurationPrettyPrinter()
    yaml_parser = ConfigurationYAMLParser()
    
    config = TranscriptionConfig(
        engine="aws-transcribe",
        output_dir="./aws-output",
        log_level="info"
    )
    
    # Format with pretty printer (minimal style for cleaner round-trip)
    formatted_yaml = pretty_printer.format_configuration(config, style="minimal")
    
    # Parse back
    parsed_dict = yaml_parser.parse_string(formatted_yaml)
    
    # Convert back to configuration object
    from pipeline.config.manager import ConfigurationManager
    config_manager = ConfigurationManager()
    parsed_config = config_manager._dict_to_config(parsed_dict)
    
    # Essential values should be preserved
    assert parsed_config.engine == "aws-transcribe"
    assert parsed_config.output_dir == "./aws-output"
    assert parsed_config.log_level == "info"


def test_round_trip_with_environment_variables():
    """
    **Property 8: YAML Round-Trip Consistency**
    Configuration with environment variable placeholders should round-trip correctly.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    # Create YAML with environment variable placeholders
    yaml_content = """
engine: openai-whisper
output_dir: ./transcripts
whisper_api:
  api_key: ${OPENAI_API_KEY}
  model: whisper-1
aws_transcribe:
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
"""
    
    # Parse the YAML
    parsed_dict = yaml_parser.parse_string(yaml_content)
    
    # Convert to configuration object (without environment substitution)
    from pipeline.config.manager import ConfigurationManager
    config_manager = ConfigurationManager()
    config = config_manager._dict_to_config(parsed_dict)
    
    # Serialize back to YAML
    serialized_yaml = yaml_parser.serialize_to_yaml(config, include_comments=False)
    
    # Parse again
    reparsed_dict = yaml_parser.parse_string(serialized_yaml)
    reparsed_config = config_manager._dict_to_config(reparsed_dict)
    
    # Core values should be preserved
    assert reparsed_config.engine == config.engine
    assert reparsed_config.output_dir == config.output_dir
    assert reparsed_config.whisper_api.model == config.whisper_api.model


def test_round_trip_preserves_data_types():
    """
    **Property 8: YAML Round-Trip Consistency**
    YAML round-trip should preserve data types (integers, floats, booleans, strings).
    """
    yaml_parser = ConfigurationYAMLParser()
    
    config = TranscriptionConfig(
        engine="local-whisper",
        output_dir="./test-output",
        log_level="debug",
        language="en"
    )
    
    # Serialize to YAML
    yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
    
    # Parse back
    parsed_dict = yaml_parser.parse_string(yaml_content)
    
    # Check data types are preserved
    assert isinstance(parsed_dict['engine'], str)
    assert isinstance(parsed_dict['output_dir'], str)
    assert isinstance(parsed_dict['log_level'], str)
    assert isinstance(parsed_dict['language'], str)
    
    assert isinstance(parsed_dict['whisper_local']['timeout'], int)
    assert isinstance(parsed_dict['whisper_local']['retry_attempts'], int)
    assert isinstance(parsed_dict['whisper_local']['retry_delay'], float)
    
    assert isinstance(parsed_dict['whisper_api']['temperature'], float)
    assert isinstance(parsed_dict['whisper_api']['timeout'], int)
    
    assert isinstance(parsed_dict['auto_selection']['prefer_local'], bool)
    assert isinstance(parsed_dict['auto_selection']['fallback_enabled'], bool)
    assert isinstance(parsed_dict['auto_selection']['priority_order'], list)


def test_round_trip_with_all_engine_types():
    """
    **Property 8: YAML Round-Trip Consistency**
    Round-trip should work correctly for all supported engine types.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    for engine_type in [e.value for e in EngineType]:
        config = TranscriptionConfig(engine=engine_type)
        
        # Serialize and parse back
        yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
        parsed_dict = yaml_parser.parse_string(yaml_content)
        
        from pipeline.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        parsed_config = config_manager._dict_to_config(parsed_dict)
        
        # Engine type should be preserved
        assert parsed_config.engine == engine_type


def test_round_trip_with_all_log_levels():
    """
    **Property 8: YAML Round-Trip Consistency**
    Round-trip should work correctly for all supported log levels.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    for log_level in [l.value for l in LogLevel]:
        config = TranscriptionConfig(log_level=log_level)
        
        # Serialize and parse back
        yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
        parsed_dict = yaml_parser.parse_string(yaml_content)
        
        from pipeline.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        parsed_config = config_manager._dict_to_config(parsed_dict)
        
        # Log level should be preserved
        assert parsed_config.log_level == log_level


def test_round_trip_with_all_whisper_models():
    """
    **Property 8: YAML Round-Trip Consistency**
    Round-trip should work correctly for all supported Whisper model sizes.
    """
    yaml_parser = ConfigurationYAMLParser()
    
    for model_size in [m.value for m in WhisperModelSize]:
        config = TranscriptionConfig()
        config.whisper_local.model = model_size
        
        # Serialize and parse back
        yaml_content = yaml_parser.serialize_to_yaml(config, include_comments=False)
        parsed_dict = yaml_parser.parse_string(yaml_content)
        
        from pipeline.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        parsed_config = config_manager._dict_to_config(parsed_dict)
        
        # Model size should be preserved
        assert parsed_config.whisper_local.model == model_size


"""
Property-based tests for engine factory instantiation.

**Property 10: Engine Factory Instantiation**
*For any* valid engine type and configuration, the EngineFactory should create 
a working adapter that conforms to the TranscriberAdapter protocol.
**Validates: Requirements 8.1, 8.2, 8.4**
"""

from hypothesis import given, strategies as st, assume, settings
from unittest.mock import patch, Mock
from pipeline.transcription.factory import TranscriptionProviderFactory
from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.config.schema import TranscriptionConfig, WhisperLocalConfig, WhisperModelSize


# Strategy for generating valid engine types (currently only whisper-local is implemented)
valid_engine_types = st.sampled_from(['local-whisper'])

# Strategy for generating valid Whisper model sizes
valid_whisper_models = st.sampled_from([m.value for m in WhisperModelSize])


@given(
    engine_type=valid_engine_types,
    model_size=valid_whisper_models
)
@settings(deadline=None)  # Disable deadline for this test
def test_factory_creates_valid_adapters(engine_type, model_size):
    """
    **Property 10: Engine Factory Instantiation**
    *For any* valid engine type and configuration, the factory should create a working adapter.
    """
    config = TranscriptionConfig(
        engine=engine_type,
        whisper_local=WhisperLocalConfig(model=model_size)
    )
    
    factory = TranscriptionProviderFactory(config)
    
    # Mock the requirements validation to avoid actual model loading
    with patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter.validate_requirements', return_value=[]), \
         patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter._load_model'):
        adapter = factory.create_engine(engine_type, config)
        
        # Adapter should implement the protocol
        assert hasattr(adapter, 'transcribe')
        assert hasattr(adapter, 'get_engine_info')
        assert hasattr(adapter, 'validate_requirements')
        assert hasattr(adapter, 'get_supported_formats')
        assert hasattr(adapter, 'estimate_cost')
        
        # All protocol methods should be callable
        assert callable(adapter.transcribe)
        assert callable(adapter.get_engine_info)
        assert callable(adapter.validate_requirements)
        assert callable(adapter.get_supported_formats)
        assert callable(adapter.estimate_cost)


@given(
    engine_type=valid_engine_types,
    model_size=valid_whisper_models
)
@settings(deadline=None)  # Disable deadline for this test
def test_factory_passes_configuration_correctly(engine_type, model_size):
    """
    **Property 10: Engine Factory Instantiation**
    *For any* valid configuration, the factory should pass the configuration to the adapter correctly.
    """
    config = TranscriptionConfig(
        engine=engine_type,
        whisper_local=WhisperLocalConfig(model=model_size)
    )
    
    factory = TranscriptionProviderFactory(config)
    
    with patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter.validate_requirements', return_value=[]), \
         patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter._load_model'):
        adapter = factory.create_engine(engine_type, config)
        
        # Configuration should be passed correctly
        if engine_type == 'local-whisper':
            assert adapter.model_name == model_size


@given(invalid_engine=st.text().filter(lambda x: x not in ['local-whisper']))
def test_factory_rejects_invalid_engines(invalid_engine):
    """
    **Property 10: Engine Factory Instantiation**
    *For any* invalid engine type, the factory should raise a clear error.
    """
    assume(invalid_engine.strip() != "")  # Skip empty strings
    
    config = TranscriptionConfig(engine=invalid_engine)
    factory = TranscriptionProviderFactory(config)
    
    with pytest.raises(ValueError, match="Unsupported engine type"):
        factory.create_engine(invalid_engine, config)


def test_factory_validates_requirements_before_creation():
    """
    **Property 10: Engine Factory Instantiation**
    *For any* engine with unmet requirements, the factory should raise a clear error.
    """
    config = TranscriptionConfig(engine='local-whisper')
    factory = TranscriptionProviderFactory(config)
    
    # Mock requirements validation to return errors
    mock_errors = ['Whisper not installed', 'Model not available']
    with patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter.validate_requirements', 
               return_value=mock_errors):
        with pytest.raises(RuntimeError, match="Engine 'local-whisper' requirements not met"):
            factory.create_engine('local-whisper', config)


def test_factory_available_engines_consistency():
    """
    **Property 10: Engine Factory Instantiation**
    *For any* engine returned by get_available_engines(), the factory should be able to create it.
    """
    config = TranscriptionConfig()
    factory = TranscriptionProviderFactory(config)
    available_engines = factory.get_available_engines()
    
    assert len(available_engines) > 0, "Factory should have at least one available engine"
    
    for engine_type in available_engines:
        # Should be able to check if engine is available
        assert factory.is_engine_available(engine_type) is True
        
        # Should be able to get engine info
        info = factory.get_engine_info(engine_type)
        assert info['engine_type'] == engine_type
        assert info['is_available'] is True


def test_factory_registration_consistency():
    """
    **Property 10: Engine Factory Instantiation**
    *For any* newly registered adapter, the factory should be able to create and use it.
    """
    config = TranscriptionConfig()
    factory = TranscriptionProviderFactory(config)
    
    # Create a mock adapter class
    mock_adapter = Mock()
    mock_adapter.validate_requirements.return_value = []
    mock_adapter_class = Mock(return_value=mock_adapter)
    
    # Register the adapter
    engine_name = 'test-engine'
    factory.register_adapter(engine_name, mock_adapter_class)
    
    # Should be available
    assert factory.is_engine_available(engine_name) is True
    assert engine_name in factory.get_available_engines()
    
    # Should be able to create it
    config = TranscriptionConfig(engine=engine_name)
    adapter = factory.create_engine(engine_name, config)
    
    assert adapter is mock_adapter
    mock_adapter_class.assert_called_once()


@given(
    engine_type=valid_engine_types,
    model_size=valid_whisper_models
)
def test_factory_engine_info_accuracy(engine_type, model_size):
    """
    **Property 10: Engine Factory Instantiation**
    *For any* available engine, get_engine_info should return accurate information.
    """
    config = TranscriptionConfig()
    factory = TranscriptionProviderFactory(config)
    
    info = factory.get_engine_info(engine_type)
    
    # Info should contain required fields
    assert 'engine_type' in info
    assert 'adapter_class' in info
    assert 'module' in info
    assert 'is_available' in info
    
    # Values should be correct
    assert info['engine_type'] == engine_type
    assert info['is_available'] is True
    assert isinstance(info['adapter_class'], str)
    assert isinstance(info['module'], str)


def test_factory_requirement_validation_consistency():
    """
    **Property 10: Engine Factory Instantiation**
    *For any* engine, validate_engine_requirements should return the same errors as the adapter.
    """
    config = TranscriptionConfig(engine='local-whisper')
    factory = TranscriptionProviderFactory(config)
    
    expected_errors = ['Test error 1', 'Test error 2']
    
    with patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter.validate_requirements', 
               return_value=expected_errors):
        # Factory validation should return the same errors
        factory_errors = factory.validate_engine_requirements('local-whisper', config)
        assert factory_errors == expected_errors
        
        # Creating the engine should fail with the same errors
        with pytest.raises(RuntimeError) as exc_info:
            factory.create_engine('local-whisper', config)
        
        error_message = str(exc_info.value)
        for expected_error in expected_errors:
            assert expected_error in error_message


@given(st.data())
@settings(deadline=None)  # Disable deadline for this test
def test_factory_handles_all_configuration_combinations(data):
    """
    **Property 10: Engine Factory Instantiation**
    *For any* valid configuration combination, the factory should handle it gracefully.
    """
    # Generate a configuration with various settings
    engine_type = data.draw(valid_engine_types)
    model_size = data.draw(valid_whisper_models)
    
    config = TranscriptionConfig(
        engine=engine_type,
        output_dir=data.draw(st.text(min_size=1, max_size=20)),
        log_level=data.draw(st.sampled_from(['debug', 'info', 'warning', 'error'])),
        whisper_local=WhisperLocalConfig(
            model=model_size,
            timeout=data.draw(st.integers(min_value=1, max_value=600)),
            retry_attempts=data.draw(st.integers(min_value=0, max_value=5))
        )
    )
    
    factory = TranscriptionProviderFactory(config)
    
    with patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter.validate_requirements', return_value=[]), \
         patch('pipeline.transcribers.adapters.local_whisper.LocalWhisperAdapter._load_model'):
        # Should be able to create adapter regardless of other configuration settings
        adapter = factory.create_engine(engine_type, config)
        
        # Adapter should have the correct model configuration
        if engine_type == 'local-whisper':
            assert adapter.model_name == model_size