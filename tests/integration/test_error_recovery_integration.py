"""
Error Recovery Integration Tests

Integration tests for error scenarios and recovery paths in the
enhanced transcription system.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cli.transcribe import transcribe
from pipeline.utils.error_messages import ErrorMessages, ErrorCategory
from pipeline.utils.logging_config import logging_config


@pytest.mark.integration
@pytest.mark.slow
class TestErrorRecoveryIntegration:
    """Integration tests for error scenarios and recovery."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.test_audio_file = "tests/assets/sample_audio.mp3"
    
    def test_missing_engine_flag_error_recovery(self, tmp_path):
        """Test error recovery for missing engine flag (breaking change)."""
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Should fail with clear error
        assert result.exit_code != 0
        
        # Should provide migration guidance
        output_lower = result.output.lower()
        migration_keywords = [
            'engine', 'required', 'breaking', 'v0.6.5',
            'local-whisper', 'openai-whisper', 'auto', 'example'
        ]
        assert any(keyword in output_lower for keyword in migration_keywords)
        
        # Should show valid examples
        assert '--engine' in result.output

    def test_invalid_engine_error_recovery(self, tmp_path):
        """Test error recovery for invalid engine selection."""
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'invalid-engine',
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Should fail with clear error
        assert result.exit_code != 0
        
        # Should list valid engines
        output_lower = result.output.lower()
        valid_engines = ['local-whisper', 'openai-whisper', 'auto']
        assert any(engine in output_lower for engine in valid_engines)

    def test_missing_source_file_error_recovery(self, tmp_path):
        """Test error recovery for missing source file."""
        nonexistent_file = str(tmp_path / 'nonexistent.mp3')
        
        result = self.runner.invoke(transcribe, [
            '--source', nonexistent_file,
            '--engine', 'local-whisper',
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Should fail with file not found error
        assert result.exit_code != 0
        
        # Should provide helpful error message
        output_lower = result.output.lower()
        file_error_keywords = ['file', 'not found', 'exist', 'path']
        assert any(keyword in output_lower for keyword in file_error_keywords)

    def test_configuration_file_error_recovery(self, tmp_path):
        """Test error recovery for invalid configuration files."""
        # Create invalid YAML configuration
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text("""
engine: local-whisper
output_dir: ./transcripts
whisper_local:
  model: base
  invalid_yaml_syntax: [unclosed bracket
""")
        
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'local-whisper',
            '--config', str(invalid_config),
            '--output-dir', str(tmp_path / 'output')
        ])
        
        # Should fail with configuration error
        assert result.exit_code != 0
        
        # Should provide YAML syntax guidance
        output_lower = result.output.lower()
        yaml_error_keywords = ['yaml', 'syntax', 'configuration', 'invalid']
        assert any(keyword in output_lower for keyword in yaml_error_keywords)
    @patch('openai.OpenAI')
    def test_api_authentication_error_recovery(self, mock_openai_class, tmp_path):
        """Test error recovery for API authentication failures."""
        # Test 1: Missing API key (clear environment)
        with patch.dict(os.environ, {}, clear=True):
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'openai-whisper',
                '--output-dir', str(tmp_path / 'output')
            ])
            
            # Should fail with authentication error
            assert result.exit_code != 0
            
            # Should provide API key setup guidance
            output_lower = result.output.lower()
            # Check for error indicators (more lenient)
            has_error = ('error' in output_lower or 'fail' in output_lower or 
                        'api' in output_lower or 'key' in output_lower or
                        'required' in output_lower or 'missing' in output_lower)
            assert has_error, f"Expected error message, got: {result.output}"
        
        # Test 2: API error during transcription
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("Invalid API key")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-invalid-key'}):
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'openai-whisper',
                '--output-dir', str(tmp_path / 'output')
            ])
            
            # Should fail with API error
            assert result.exit_code != 0
            
            # Should provide helpful error message
            assert 'api' in result.output.lower() or 'error' in result.output.lower()

    def test_output_directory_permission_error_recovery(self, tmp_path):
        """Test error recovery for output directory permission issues."""
        # Create a directory with restricted permissions
        restricted_dir = tmp_path / 'restricted'
        restricted_dir.mkdir()
        
        # Make directory read-only (simulate permission error)
        if os.name != 'nt':  # Skip on Windows due to different permission model
            os.chmod(restricted_dir, 0o444)
            
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'local-whisper',
                '--model', 'tiny',
                '--output-dir', str(restricted_dir)
            ])
            
            # Should fail with permission error
            assert result.exit_code != 0
            
            # Should provide permission guidance
            output_lower = result.output.lower()
            permission_keywords = ['permission', 'denied', 'access', 'directory']
            assert any(keyword in output_lower for keyword in permission_keywords)
            
            # Restore permissions for cleanup
            os.chmod(restricted_dir, 0o755)

    def test_engine_requirement_error_recovery(self, tmp_path):
        """Test error recovery when engine requirements are not met."""
        # This test simulates when Whisper is not installed
        with patch('pipeline.transcribers.factory.EngineFactory.validate_engine_requirements') as mock_validate:
            mock_validate.return_value = [
                "Whisper is not installed",
                "Run: pip install openai-whisper"
            ]
            
            result = self.runner.invoke(transcribe, [
                '--source', self.test_audio_file,
                '--engine', 'local-whisper',
                '--output-dir', str(tmp_path / 'output')
            ])
            
            # Should fail with engine requirement error
            assert result.exit_code != 0
            
            # Should provide installation guidance
            output_lower = result.output.lower()
            requirement_keywords = ['install', 'whisper', 'pip', 'requirement', 'available']
            assert any(keyword in output_lower for keyword in requirement_keywords)

    def test_auto_selection_fallback_recovery(self, tmp_path):
        """Test auto-selection fallback when preferred engine fails."""
        # Mock scenario where local engine fails but API is available
        with patch('pipeline.transcribers.auto_selector.AutoSelector.check_local_whisper_availability') as mock_local:
            with patch('pipeline.transcribers.auto_selector.AutoSelector.check_api_key_availability') as mock_api:
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                    mock_local.return_value = False  # Local not available
                    mock_api.return_value = True     # API available
                    
                    # Mock successful API transcription
                    with patch('pipeline.transcribers.adapters.openai_whisper.openai') as mock_openai:
                        mock_response = MagicMock()
                        mock_response.text = "Test transcription"
                        mock_openai.Audio.transcribe.return_value = mock_response
                        
                        result = self.runner.invoke(transcribe, [
                            '--source', self.test_audio_file,
                            '--engine', 'auto',
                            '--output-dir', str(tmp_path / 'output')
                        ])
                        
                        # Should succeed with fallback
                        assert result.exit_code == 0
                        assert "Auto-selected engine:" in result.output
                        assert "openai-whisper" in result.output.lower()

    def test_comprehensive_error_message_formatting(self, tmp_path):
        """Test that error messages are properly formatted and helpful."""
        # Test various error scenarios to ensure consistent formatting
        
        # Test 1: Configuration error formatting
        result = self.runner.invoke(transcribe, [
            '--source', self.test_audio_file,
            '--engine', 'invalid-engine'
        ])
        
        assert result.exit_code != 0
        # Should have structured error message
        assert any(char in result.output for char in ['•', '-', '*'])  # Bullet points
        assert 'Suggestions:' in result.output or 'Options:' in result.output
        
        # Test 2: File error formatting
        result = self.runner.invoke(transcribe, [
            '--source', 'nonexistent.mp3',
            '--engine', 'local-whisper'
        ])
        
        assert result.exit_code != 0
        # Should provide actionable suggestions
        suggestions_present = any(word in result.output.lower() for word in [
            'check', 'verify', 'ensure', 'try', 'use'
        ])
        assert suggestions_present