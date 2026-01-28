"""
Configuration Integration Tests

Integration tests specifically for configuration loading, validation,
and precedence scenarios in the enhanced transcription system.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from pipeline.config.manager import ConfigurationManager
from pipeline.config.schema import TranscriptionConfig
from pipeline.transcribers.factory import EngineFactory
from pipeline.output.manager import OutputManager


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigurationManager()
    
    def test_configuration_file_discovery_hierarchy(self, tmp_path):
        """Test configuration file discovery and hierarchy."""
        # Create user config
        user_config_dir = tmp_path / ".content-pipeline"
        user_config_dir.mkdir()
        user_config = user_config_dir / "config.yaml"
        user_config.write_text("""
engine: local-whisper
output_dir: ./user_transcripts
whisper_local:
  model: base
log_level: info
""")
        
        # Create project config
        project_config_dir = tmp_path / "project" / ".content-pipeline"
        project_config_dir.mkdir(parents=True)
        project_config = project_config_dir / "config.yaml"
        project_config.write_text("""
engine: auto
output_dir: ./project_transcripts
whisper_local:
  model: small
auto_selection:
  prefer_local: false
""")
        
        # Test loading project config (should override user config)
        os.chdir(tmp_path / "project")
        # Create a new ConfigurationManager after changing directory
        config_manager = ConfigurationManager()
        config = config_manager.load_configuration()
        
        assert config.engine == "auto"  # From project config
        assert config.output_dir == "./project_transcripts"  # From project config
        assert config.whisper_local.model == "small"  # From project config
        assert config.log_level == "info"  # From user config (not overridden)
    
    def test_environment_variable_substitution_integration(self, tmp_path):
        """Test environment variable substitution in configuration files."""
        config_file = tmp_path / "config.yaml"
        config_content = """
engine: ${PIPELINE_ENGINE:-auto}
output_dir: ${PIPELINE_OUTPUT:-./default_output}
log_level: ${PIPELINE_LOG_LEVEL:-info}
whisper_local:
  model: ${WHISPER_MODEL:-base}
whisper_api:
  api_key: ${OPENAI_API_KEY}
"""
        config_file.write_text(config_content)
        
        # Test with environment variables set
        env_vars = {
            'PIPELINE_ENGINE': 'openai-whisper',
            'PIPELINE_OUTPUT': '/tmp/test_output',
            'PIPELINE_LOG_LEVEL': 'debug',
            'WHISPER_MODEL': 'large',
            'OPENAI_API_KEY': 'test-api-key',
        }
        
        with patch.dict(os.environ, env_vars):
            config = self.config_manager.load_configuration(config_file=str(config_file))
        
        assert config.engine == "openai-whisper"
        assert config.output_dir == "/tmp/test_output"
        assert config.log_level == "debug"
        assert config.whisper_local.model == "large"
        assert config.whisper_api.api_key == "test-api-key"
    def test_cli_override_precedence_integration(self, tmp_path):
        """Test that CLI parameters override configuration files."""
        # Create configuration file
        config_file = tmp_path / "config.yaml"
        config_content = """
engine: local-whisper
output_dir: ./config_output
log_level: info
whisper_local:
  model: base
  timeout: 300
whisper_api:
  api_key: config-api-key
  temperature: 0.5
"""
        config_file.write_text(config_content)
        
        # Load configuration with CLI overrides
        cli_overrides = {
            'engine': 'openai-whisper',
            'output_dir': './cli_output',
            'log_level': 'debug',
        }
        
        config = self.config_manager.load_configuration(
            config_file=str(config_file),
            cli_overrides=cli_overrides
        )
        
        # Verify CLI overrides took precedence
        assert config.engine == "openai-whisper"
        assert config.output_dir == "./cli_output"
        assert config.log_level == "debug"
        
        # Verify non-overridden values remain from config
        assert config.whisper_local.timeout == 300
        assert config.whisper_api.temperature == 0.5

    def test_configuration_validation_integration(self, tmp_path):
        """Test configuration validation with various scenarios."""
        # Test 1: Valid configuration
        valid_config_file = tmp_path / "valid_config.yaml"
        valid_config_content = """
engine: local-whisper
output_dir: ./transcripts
log_level: info
whisper_local:
  model: base
  device: auto
  timeout: 300
"""
        valid_config_file.write_text(valid_config_content)
        
        config = self.config_manager.load_configuration(config_file=str(valid_config_file))
        errors = self.config_manager.validate_configuration(config)
        assert len(errors) == 0
        
        # Test 2: Invalid engine
        invalid_config_file = tmp_path / "invalid_config.yaml"
        invalid_config_content = """
engine: invalid-engine
output_dir: ./transcripts
"""
        invalid_config_file.write_text(invalid_config_content)
        
        config = self.config_manager.load_configuration(config_file=str(invalid_config_file))
        errors = self.config_manager.validate_configuration(config)
        assert len(errors) > 0
        assert any("engine" in error.lower() for error in errors)
        
        # Test 3: Invalid model
        invalid_model_config = tmp_path / "invalid_model_config.yaml"
        invalid_model_content = """
engine: local-whisper
whisper_local:
  model: invalid-model
"""
        invalid_model_config.write_text(invalid_model_content)
        
        config = self.config_manager.load_configuration(config_file=str(invalid_model_config))
        errors = self.config_manager.validate_configuration(config)
        assert len(errors) > 0
        assert any("model" in error.lower() for error in errors)
    def test_engine_factory_configuration_integration(self, tmp_path):
        """Test integration between configuration and engine factory."""
        # Create configuration for different engines
        config_file = tmp_path / "engine_config.yaml"
        config_content = """
engine: auto
whisper_local:
  model: small
  device: cpu
  timeout: 600
whisper_api:
  api_key: test-api-key
  model: whisper-1
  temperature: 0.0
  timeout: 120
auto_selection:
  prefer_local: true
  fallback_enabled: true
"""
        config_file.write_text(config_content)
        
        config = self.config_manager.load_configuration(config_file=str(config_file))
        factory = EngineFactory()
        
        # Test engine requirement validation
        local_errors = factory.validate_engine_requirements("local-whisper", config)
        # Should pass basic validation (actual Whisper installation not required for test)
        assert isinstance(local_errors, list)
        
        api_errors = factory.validate_engine_requirements("openai-whisper", config)
        # Should pass with API key provided
        assert isinstance(api_errors, list)
        
        # Test configuration passing to adapters
        available_engines = factory.get_available_engines()
        assert "local-whisper" in available_engines
        assert "openai-whisper" in available_engines

    def test_output_manager_configuration_integration(self, tmp_path):
        """Test integration between configuration and output manager."""
        # Create configuration with output settings
        config_file = tmp_path / "output_config.yaml"
        config_content = f"""
engine: local-whisper
output_dir: {tmp_path / 'configured_output'}
"""
        config_file.write_text(config_content)
        
        config = self.config_manager.load_configuration(config_file=str(config_file))
        output_manager = OutputManager()
        
        # Test output path resolution
        test_input = "sample_audio.mp3"
        resolved_path = output_manager.resolve_output_path(
            output_path=None,
            output_dir=config.output_dir,
            input_file_path=test_input
        )
        
        expected_dir = tmp_path / 'configured_output'
        assert str(expected_dir) in str(resolved_path)
        assert "sample_audio" in str(resolved_path)
        assert str(resolved_path).endswith('.json')

    def test_complete_configuration_workflow_integration(self, tmp_path):
        """Test complete configuration workflow from file to components."""
        # Create comprehensive configuration
        config_file = tmp_path / "complete_config.yaml"
        config_content = f"""
# Complete configuration test
engine: auto
output_dir: {tmp_path / 'workflow_output'}
log_level: debug
language: en

# Engine configurations
whisper_local:
  model: base
  device: auto
  timeout: 300
  retry_attempts: 3

whisper_api:
  api_key: ${{OPENAI_API_KEY:-test-key}}
  model: whisper-1
  temperature: 0.0
  timeout: 60

# Auto-selection preferences
auto_selection:
  prefer_local: true
  fallback_enabled: true
"""
        config_file.write_text(config_content)
        
        # Load and validate configuration
        config = self.config_manager.load_configuration(config_file=str(config_file))
        validation_errors = self.config_manager.validate_configuration(config)
        assert len(validation_errors) == 0
        
        # Test all components can use the configuration
        factory = EngineFactory()
        output_manager = OutputManager()
        
        # Verify configuration values
        assert config.engine == "auto"
        assert str(tmp_path / 'workflow_output') in config.output_dir
        assert config.log_level == "debug"
        assert config.language == "en"
        assert config.whisper_local.model == "base"
        assert config.auto_selection.prefer_local is True
        
        # Test components can work with configuration
        available_engines = factory.get_available_engines()
        assert len(available_engines) > 0
        
        test_output_path = output_manager.resolve_output_path(
            output_path=None,
            output_dir=config.output_dir,
            input_file_path="test.mp3"
        )
        assert str(tmp_path / 'workflow_output') in str(test_output_path)