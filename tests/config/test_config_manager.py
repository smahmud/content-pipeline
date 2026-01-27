"""
Unit tests for Configuration Manager.

Tests the configuration loading, validation, and environment variable integration.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from pipeline.config.manager import ConfigurationManager
from pipeline.config.schema import TranscriptionConfig, EngineType, LogLevel


class TestConfigurationManager:
    """Test suite for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
    
    def test_load_default_configuration(self):
        """Test loading default configuration with no files or overrides."""
        config = self.config_manager.load_configuration()
        
        assert config.engine == EngineType.AUTO.value
        assert config.output_dir == "./transcripts"
        assert config.log_level == LogLevel.INFO.value
        assert config.whisper_local.model == "base"
        assert config.whisper_api.model == "whisper-1"
        assert config.aws_transcribe.region == "us-east-1"
    
    def test_load_configuration_with_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
engine: whisper-local
output_dir: ./custom-output
log_level: debug
whisper_local:
  model: large
  timeout: 600
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(yaml_content)
            
            config = self.config_manager.load_configuration(config_file=str(config_file))
            
            assert config.engine == "whisper-local"
            assert config.output_dir == "./custom-output"
            assert config.log_level == "debug"
            assert config.whisper_local.model == "large"
            assert config.whisper_local.timeout == 600
    
    def test_cli_overrides_take_precedence(self):
        """Test that CLI overrides have highest precedence."""
        yaml_content = """
engine: whisper-local
output_dir: ./yaml-output
log_level: debug
"""
        
        cli_overrides = {
            'engine': 'whisper-api',
            'output_dir': './cli-output',
            'whisper_api': {'model': 'whisper-1-custom'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(yaml_content)
            
            config = self.config_manager.load_configuration(
                config_file=str(config_file),
                cli_overrides=cli_overrides
            )
            
            # CLI overrides should win
            assert config.engine == "whisper-api"
            assert config.output_dir == "./cli-output"
            assert config.whisper_api.model == "whisper-1-custom"
            
            # YAML values should be preserved where not overridden
            assert config.log_level == "debug"
    
    @patch.dict(os.environ, {
        'CONTENT_PIPELINE_DEFAULT_ENGINE': 'aws-transcribe',
        'CONTENT_PIPELINE_OUTPUT_DIR': './env-output',
        'CONTENT_PIPELINE_LOG_LEVEL': 'debug',
        'WHISPER_LOCAL_MODEL': 'large',
        'OPENAI_API_KEY': 'sk-test-key',
        'WHISPER_API_MODEL': 'whisper-1-custom',
        'AWS_ACCESS_KEY_ID': 'test-access-key',
        'AWS_SECRET_ACCESS_KEY': 'test-secret-key',
        'AWS_DEFAULT_REGION': 'us-west-2'
    })
    def test_environment_variables_loaded(self):
        """Test that environment variables are loaded correctly."""
        config = self.config_manager.load_configuration()
        
        assert config.engine == "aws-transcribe"
        assert config.output_dir == "./env-output"
        assert config.log_level == "debug"
        assert config.whisper_local.model == "large"
        assert config.whisper_api.api_key == "sk-test-key"
        assert config.whisper_api.model == "whisper-1-custom"
        assert config.aws_transcribe.access_key_id == "test-access-key"
        assert config.aws_transcribe.secret_access_key == "test-secret-key"
        assert config.aws_transcribe.region == "us-west-2"
    
    def test_environment_variable_substitution(self):
        """Test ${VAR} and ${VAR:-default} substitution."""
        config_dict = {
            'api_key': '${TEST_API_KEY}',
            'output_dir': '${TEST_OUTPUT_DIR:-./default-output}',
            'missing_var': '${MISSING_VAR:-fallback-value}',
            'nested': {
                'value': '${NESTED_VAR:-nested-default}'
            }
        }
        
        with patch.dict(os.environ, {
            'TEST_API_KEY': 'actual-key',
            'NESTED_VAR': 'actual-nested'
        }):
            result = self.config_manager.substitute_environment_variables(config_dict)
            
            assert result['api_key'] == 'actual-key'
            assert result['output_dir'] == './default-output'  # Uses default
            assert result['missing_var'] == 'fallback-value'  # Uses default
            assert result['nested']['value'] == 'actual-nested'
    
    def test_environment_variable_substitution_missing_required(self):
        """Test that missing required environment variables raise error."""
        config_dict = {
            'api_key': '${REQUIRED_VAR}'  # No default provided
        }
        
        with pytest.raises(ValueError, match="Required environment variable 'REQUIRED_VAR' is not set"):
            self.config_manager.substitute_environment_variables(config_dict)
    
    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises appropriate error."""
        invalid_yaml = """
engine: whisper-local
  invalid: yaml: structure
    - missing proper indentation
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid_config.yaml"
            config_file.write_text(invalid_yaml)
            
            with pytest.raises(ValueError, match="YAML parsing error"):
                self.config_manager.load_configuration(config_file=str(config_file))
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = TranscriptionConfig()
        errors = self.config_manager.validate_configuration(valid_config)
        assert errors == []
        
        # Invalid configuration
        invalid_config = TranscriptionConfig(
            engine="invalid-engine",
            log_level="invalid-level"
        )
        errors = self.config_manager.validate_configuration(invalid_config)
        assert len(errors) > 0
        assert any("Invalid engine" in error for error in errors)
        assert any("Invalid log_level" in error for error in errors)
    
    def test_save_configuration(self):
        """Test saving configuration to YAML file."""
        config = TranscriptionConfig(
            engine="whisper-local",
            output_dir="./test-output",
            log_level="debug"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "saved_config.yaml"
            
            self.config_manager.save_configuration(config, str(config_file))
            
            # Verify file was created and contains expected content
            assert config_file.exists()
            
            content = config_file.read_text()
            assert "engine: whisper-local" in content
            assert "output_dir: ./test-output" in content
            assert "log_level: debug" in content
            assert "Content Pipeline Configuration v0.6.5" in content
    
    def test_generate_default_config(self):
        """Test generating default configuration YAML."""
        yaml_content = self.config_manager.generate_default_config()
        
        assert "Content Pipeline Configuration v0.6.5" in yaml_content
        assert "engine: auto" in yaml_content
        assert "output_dir: ./transcripts" in yaml_content
        assert "log_level: info" in yaml_content
        assert "whisper_local:" in yaml_content
        assert "whisper_api:" in yaml_content
        assert "aws_transcribe:" in yaml_content
        assert "${OPENAI_API_KEY}" in yaml_content
        assert "${AWS_ACCESS_KEY_ID}" in yaml_content
    
    def test_config_merge_precedence(self):
        """Test that configuration merging follows correct precedence."""
        # Create temporary config files
        user_config = """
engine: whisper-local
output_dir: ./user-output
whisper_local:
  model: small
"""
        
        project_config = """
engine: whisper-api
log_level: debug
whisper_local:
  timeout: 600
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            user_path = Path(temp_dir) / "user.yaml"
            project_path = Path(temp_dir) / "project.yaml"
            
            user_path.write_text(user_config)
            project_path.write_text(project_config)
            
            # Mock the config paths
            with patch.object(self.config_manager, 'user_config_path', user_path), \
                 patch.object(self.config_manager, 'project_config_path', project_path):
                
                cli_overrides = {'log_level': 'error'}
                
                config = self.config_manager.load_configuration(cli_overrides=cli_overrides)
                
                # CLI override should win
                assert config.log_level == "error"
                
                # Project config should override user config
                assert config.engine == "whisper-api"
                
                # User config should be preserved where not overridden
                assert config.output_dir == "./user-output"
                
                # Nested configs should merge properly
                assert config.whisper_local.model == "small"  # From user
                assert config.whisper_local.timeout == 600    # From project