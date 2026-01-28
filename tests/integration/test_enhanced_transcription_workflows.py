"""
Enhanced Transcription Integration Tests

Integration tests for the v0.6.5 enhanced transcription system covering:
- End-to-end transcription with each engine type
- Configuration loading from all sources
- Error scenarios and recovery paths
- Complete CLI workflows with breaking changes
- Output path management and file creation
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

# Import CLI components
from cli.transcribe import transcribe
from pipeline.config.manager import ConfigurationManager
from pipeline.transcribers.factory import EngineFactory
from pipeline.transcribers.auto_selector import AutoSelector
from pipeline.output.manager import OutputManager


@pytest.mark.integration
class TestEnhancedTranscriptionWorkflows:
    """Integration tests for complete transcription workflows."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.runner = CliRunner()
        self.test_audio_file = "tests/assets/sample_audio.mp3"
        
    def test_complete_whisper_local_workflow(self, tmp_path):
        """Test complete workflow with local-whisper engine."""
        # Create test configuration
        config_file = tmp_path / "config.yaml"
        config_content = """
engine: local-whisper
output_dir: ./test_output
whisper_local:
  model: tiny
  timeout: 60
"""
        config_file.write_text(config_content)
        
        # Run transcription command
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--config', str(config_file),
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Verify successful execution
        assert result.exit_code == 0
        assert "Transcription completed successfully" in result.output
        
        # Verify output file was created
        output_files = list((tmp_path / 'output').glob('*.json'))
        assert len(output_files) == 1
        
        # Verify output content
        with open(output_files[0]) as f:
            transcript_data = json.load(f)
            assert 'metadata' in transcript_data
            assert 'transcript' in transcript_data
            assert transcript_data['metadata']['engine'] == 'whisper'
    @patch('pipeline.transcribers.adapters.whisper_api.openai')
    def test_complete_whisper_api_workflow(self, mock_openai, tmp_path):
        """Test complete workflow with openai-whisper engine."""
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.text = "This is a test transcription."
        mock_openai.Audio.transcribe.return_value = mock_response
        
        # Set API key environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'openai-whisper',
                '--output-dir', str(tmp_path / 'output')
            ])
        
        # Verify successful execution
        assert result.exit_code == 0
        assert "Transcription completed successfully" in result.output
        
        # Verify API was called
        mock_openai.Audio.transcribe.assert_called_once()
        
        # Verify output file was created
        output_files = list((tmp_path / 'output').glob('*.json'))
        assert len(output_files) == 1

    def test_auto_engine_selection_workflow(self, tmp_path):
        """Test complete workflow with auto engine selection."""
        # Create configuration favoring local engine
        config_file = tmp_path / "config.yaml"
        config_content = """
engine: auto
auto_prefer_local: true
auto_fallback_enabled: true
whisper_local:
  model: tiny
"""
        config_file.write_text(config_content)
        
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'auto',
            '--config', str(config_file),
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Verify successful execution
        assert result.exit_code == 0
        assert "Auto-selected engine:" in result.output
        assert "Transcription completed successfully" in result.output
    def test_configuration_loading_hierarchy(self, tmp_path):
        """Test configuration loading from multiple sources with proper precedence."""
        # Create user config
        user_config = tmp_path / "user_config.yaml"
        user_config.write_text("""
engine: local-whisper
output_dir: ./user_output
whisper_local:
  model: base
""")
        
        # Create project config
        project_config = tmp_path / "project_config.yaml"
        project_config.write_text("""
engine: auto
output_dir: ./project_output
whisper_local:
  model: small
""")
        
        # Test that CLI flags override configuration
        with patch.dict(os.environ, {'CONTENT_PIPELINE_OUTPUT_DIR': str(tmp_path / 'env_output')}):
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'local-whisper',  # CLI override
                '--model', 'tiny',  # CLI override
                '--config', str(project_config),
                '--output-dir', str(tmp_path / 'cli_output')  # CLI override
            ])
        
        assert result.exit_code == 0
        
        # Verify CLI output directory was used
        output_files = list((tmp_path / 'cli_output').glob('*.json'))
        assert len(output_files) == 1

    def test_environment_variable_integration(self, tmp_path):
        """Test environment variable integration and substitution."""
        # Create config with environment variable substitution
        config_file = tmp_path / "config.yaml"
        config_content = f"""
engine: local-whisper
output_dir: ${{TEST_OUTPUT_DIR:-./default_output}}
log_level: ${{TEST_LOG_LEVEL:-info}}
whisper_local:
  model: ${{TEST_MODEL:-tiny}}
"""
        config_file.write_text(config_content)
        
        # Set environment variables
        env_vars = {
            'TEST_OUTPUT_DIR': str(tmp_path / 'env_output'),
            'TEST_LOG_LEVEL': 'debug',
            'TEST_MODEL': 'base'
        }
        
        with patch.dict(os.environ, env_vars):
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'local-whisper',
                '--config', str(config_file)
            ])
        
        assert result.exit_code == 0
        
        # Verify environment variable output directory was used
        output_files = list((tmp_path / 'env_output').glob('*.json'))
        assert len(output_files) == 1
    def test_output_path_management_scenarios(self, tmp_path):
        """Test various output path management scenarios."""
        # Test 1: Absolute output path
        absolute_output = tmp_path / "absolute_transcript.json"
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--output', str(absolute_output)
        ])
        
        assert result.exit_code == 0
        assert absolute_output.exists()
        
        # Test 2: Relative output path with output directory
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--output-dir', str(tmp_path / 'relative_dir')
        ])
        
        assert result.exit_code == 0
        output_files = list((tmp_path / 'relative_dir').glob('*.json'))
        assert len(output_files) == 1
        
        # Test 3: Directory creation
        deep_dir = tmp_path / 'deep' / 'nested' / 'directory'
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--output-dir', str(deep_dir)
        ])
        
        assert result.exit_code == 0
        assert deep_dir.exists()
        output_files = list(deep_dir.glob('*.json'))
        assert len(output_files) == 1

    def test_error_scenarios_and_recovery(self, tmp_path):
        """Test error scenarios and recovery paths."""
        # Test 1: Missing engine flag (breaking change)
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file
        ])
        
        assert result.exit_code != 0
        assert "engine" in result.output.lower() or "required" in result.output.lower()
        
        # Test 2: Invalid engine type
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'invalid-engine'
        ])
        
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()
        
        # Test 3: Missing source file
        result = self.runner.invoke(transcribe, [
            '--source', 'nonexistent_file.mp3',
            '--engine', 'local-whisper'
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "file" in result.output.lower()
    def test_logging_and_progress_reporting(self, tmp_path):
        """Test logging configuration and progress reporting."""
        # Test debug logging
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--log-level', 'debug',
            '--output-dir', str(tmp_path / 'debug_output')
        ])
        
        assert result.exit_code == 0
        # Debug output should contain detailed information
        assert "DEBUG" in result.output or "Configuration Details" in result.output
        
        # Test info logging (default)
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--model', 'tiny',
            '--log-level', 'info',
            '--output-dir', str(tmp_path / 'info_output')
        ])
        
        assert result.exit_code == 0
        # Should contain progress information
        assert "Loading configuration" in result.output or "Transcribing" in result.output

    def test_breaking_change_migration_guidance(self, tmp_path):
        """Test breaking change migration guidance."""
        # Test missing engine flag provides migration guidance
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--output-dir', str(tmp_path / 'output')
        ])
        
        assert result.exit_code != 0
        # Should provide migration guidance
        migration_indicators = [
            "breaking change", "v0.6.5", "required", "engine",
            "local-whisper", "openai-whisper", "auto", "example"
        ]
        output_lower = result.output.lower()
        assert any(indicator in output_lower for indicator in migration_indicators)

    @patch('pipeline.transcribers.adapters.whisper_api.openai')
    def test_api_authentication_scenarios(self, mock_openai, tmp_path):
        """Test API authentication scenarios."""
        # Test 1: Missing API key
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'openai-whisper',
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Should fail with authentication error
        assert result.exit_code != 0
        assert "api" in result.output.lower() or "key" in result.output.lower()
        
        # Test 2: Valid API key via CLI flag
        mock_response = MagicMock()
        mock_response.text = "Test transcription"
        mock_openai.Audio.transcribe.return_value = mock_response
        
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'openai-whisper',
            '--api-key', 'test-key',
            '--output-dir', str(tmp_path / 'output')
        ])
        
        assert result.exit_code == 0