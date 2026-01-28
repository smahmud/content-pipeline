"""
Unit tests for Enhanced YAML Parser.

Tests the YAML parsing with validation, error reporting, and serialization.
"""

import tempfile
import pytest
from pathlib import Path

from pipeline.config.yaml_parser import ConfigurationYAMLParser, YAMLParsingError
from pipeline.config.schema import TranscriptionConfig


class TestConfigurationYAMLParser:
    """Test suite for ConfigurationYAMLParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ConfigurationYAMLParser()
    
    def test_parse_valid_yaml_file(self):
        """Test parsing a valid YAML configuration file."""
        yaml_content = """
engine: local-whisper
output_dir: ./test-output
log_level: debug
whisper_local:
  model: large
  timeout: 600
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(yaml_content)
            
            result = self.parser.parse_file(config_file)
            
            assert result['engine'] == 'local-whisper'
            assert result['output_dir'] == './test-output'
            assert result['log_level'] == 'debug'
            assert result['whisper_local']['model'] == 'large'
            assert result['whisper_local']['timeout'] == 600
    
    def test_parse_invalid_yaml_file_with_line_info(self):
        """Test that invalid YAML provides line number information."""
        invalid_yaml = """
engine: local-whisper
  invalid: yaml: structure
    - missing proper indentation
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid_config.yaml"
            config_file.write_text(invalid_yaml)
            
            with pytest.raises(YAMLParsingError) as exc_info:
                self.parser.parse_file(config_file)
            
            error_msg = str(exc_info.value)
            assert "Line" in error_msg
            assert str(config_file) in error_msg
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file raises appropriate error."""
        nonexistent_file = Path("/nonexistent/config.yaml")
        
        with pytest.raises(YAMLParsingError) as exc_info:
            self.parser.parse_file(nonexistent_file)
        
        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert str(nonexistent_file) in error_msg
    
    def test_parse_string_valid_yaml(self):
        """Test parsing valid YAML from string."""
        yaml_content = """
engine: openai-whisper
output_dir: ./api-output
"""
        
        result = self.parser.parse_string(yaml_content)
        
        assert result['engine'] == 'openai-whisper'
        assert result['output_dir'] == './api-output'
    
    def test_parse_string_invalid_yaml(self):
        """Test parsing invalid YAML from string provides line info."""
        invalid_yaml = """
engine: local-whisper
  invalid: yaml: structure
"""
        
        with pytest.raises(YAMLParsingError) as exc_info:
            self.parser.parse_string(invalid_yaml)
        
        error_msg = str(exc_info.value)
        assert "Line" in error_msg
    
    def test_validate_configuration_structure_valid(self):
        """Test validation of valid configuration structure."""
        config_dict = {
            'engine': 'local-whisper',
            'output_dir': './output',
            'whisper_local': {
                'model': 'base',
                'timeout': 300
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        assert errors == []
    
    def test_validate_configuration_structure_unknown_keys(self):
        """Test validation catches unknown configuration keys."""
        config_dict = {
            'engine': 'local-whisper',
            'unknown_key': 'value',
            'whisper_local': {
                'model': 'base',
                'unknown_nested_key': 'value'
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        
        assert len(errors) >= 2
        assert any('unknown_key' in error for error in errors)
        assert any('unknown_nested_key' in error for error in errors)
    
    def test_validate_whisper_local_config_invalid_types(self):
        """Test validation of whisper_local with invalid types."""
        config_dict = {
            'whisper_local': {
                'timeout': 'not-a-number',
                'retry_attempts': 'not-an-integer'
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        
        assert len(errors) >= 2
        assert any('timeout must be a number' in error for error in errors)
        assert any('retry_attempts must be an integer' in error for error in errors)
    
    def test_validate_whisper_api_config_invalid_types(self):
        """Test validation of whisper_api with invalid types."""
        config_dict = {
            'whisper_api': {
                'temperature': 'not-a-number',
                'timeout': 'not-a-number'
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        
        assert len(errors) >= 2
        assert any('temperature must be a number' in error for error in errors)
        assert any('timeout must be a number' in error for error in errors)
    
    def test_validate_auto_selection_invalid_priority_order(self):
        """Test validation of auto_selection with invalid priority order."""
        config_dict = {
            'auto_selection': {
                'priority_order': ['local-whisper', 'invalid-engine', 'openai-whisper']
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        
        assert len(errors) >= 1
        assert any('Invalid engine in priority_order' in error for error in errors)
        assert any('invalid-engine' in error for error in errors)
    
    def test_validate_auto_selection_non_list_priority_order(self):
        """Test validation of auto_selection with non-list priority order."""
        config_dict = {
            'auto_selection': {
                'priority_order': 'not-a-list'
            }
        }
        
        errors = self.parser.validate_configuration_structure(config_dict)
        
        assert len(errors) >= 1
        assert any('priority_order must be a list' in error for error in errors)
    
    def test_serialize_to_yaml_with_comments(self):
        """Test serialization to YAML with comments."""
        config = TranscriptionConfig(
            engine='local-whisper',
            output_dir='./test-output',
            log_level='debug'
        )
        
        yaml_content = self.parser.serialize_to_yaml(config, include_comments=True)
        
        assert 'Content Pipeline Configuration v0.6.5' in yaml_content
        assert 'engine: local-whisper' in yaml_content
        assert 'output_dir: ./test-output' in yaml_content
        assert 'log_level: debug' in yaml_content
        assert '# Default transcription engine' in yaml_content
        assert '${OPENAI_API_KEY}' in yaml_content
    
    def test_serialize_to_yaml_without_comments(self):
        """Test serialization to YAML without comments."""
        config = TranscriptionConfig(
            engine='openai-whisper',
            output_dir='./api-output'
        )
        
        yaml_content = self.parser.serialize_to_yaml(config, include_comments=False)
        
        assert 'engine: openai-whisper' in yaml_content
        assert 'output_dir: ./api-output' in yaml_content
        assert '# Content Pipeline Configuration' not in yaml_content
        assert '# Default transcription engine' not in yaml_content
    
    def test_validate_and_parse_file_success(self):
        """Test combined validation and parsing of valid file."""
        yaml_content = """
engine: local-whisper
output_dir: ./test-output
whisper_local:
  model: base
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(yaml_content)
            
            config_dict, errors = self.parser.validate_and_parse_file(config_file)
            
            assert errors == []
            assert config_dict['engine'] == 'local-whisper'
            assert config_dict['output_dir'] == './test-output'
    
    def test_validate_and_parse_file_with_validation_errors(self):
        """Test combined validation and parsing with validation errors."""
        yaml_content = """
engine: local-whisper
unknown_key: value
whisper_local:
  model: base
  unknown_nested: value
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(yaml_content)
            
            config_dict, errors = self.parser.validate_and_parse_file(config_file)
            
            assert len(errors) >= 2
            assert config_dict['engine'] == 'local-whisper'
            assert any('unknown_key' in error for error in errors)
    
    def test_empty_yaml_file(self):
        """Test parsing empty YAML file returns empty dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "empty_config.yaml"
            config_file.write_text("")
            
            result = self.parser.parse_file(config_file)
            assert result == {}
    
    def test_yaml_with_only_comments(self):
        """Test parsing YAML file with only comments."""
        yaml_content = """
# This is a comment
# Another comment
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "comments_only.yaml"
            config_file.write_text(yaml_content)
            
            result = self.parser.parse_file(config_file)
            assert result == {}