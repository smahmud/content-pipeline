"""
Unit tests for Environment Variable utilities.

Tests the centralized environment variable definitions and validation.
"""

import os
import pytest
from unittest.mock import patch

from pipeline.config.environment import EnvironmentVariables


class TestEnvironmentVariables:
    """Test suite for EnvironmentVariables utilities."""
    
    def test_get_all_variables(self):
        """Test that all environment variables are listed."""
        variables = EnvironmentVariables.get_all_variables()
        
        expected_variables = [
            'CONTENT_PIPELINE_DEFAULT_ENGINE',
            'CONTENT_PIPELINE_OUTPUT_DIR',
            'CONTENT_PIPELINE_LOG_LEVEL',
            'WHISPER_LOCAL_MODEL',
            'OPENAI_API_KEY',
            'WHISPER_API_MODEL',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION'
        ]
        
        assert set(variables) == set(expected_variables)
        assert len(variables) == len(expected_variables)
    
    def test_get_variable_documentation(self):
        """Test that documentation is provided for all variables."""
        docs = EnvironmentVariables.get_variable_documentation()
        variables = EnvironmentVariables.get_all_variables()
        
        # All variables should have documentation
        assert set(docs.keys()) == set(variables)
        
        # All documentation should be non-empty strings
        for var, doc in docs.items():
            assert isinstance(doc, str)
            assert len(doc.strip()) > 0
    
    @patch.dict(os.environ, {
        'CONTENT_PIPELINE_DEFAULT_ENGINE': 'invalid-engine',
        'CONTENT_PIPELINE_LOG_LEVEL': 'invalid-level',
        'WHISPER_LOCAL_MODEL': 'invalid-model'
    })
    def test_validate_environment_setup_with_errors(self):
        """Test validation with invalid environment variable values."""
        warnings, errors = EnvironmentVariables.validate_environment_setup()
        
        assert len(errors) == 3
        assert any('Invalid CONTENT_PIPELINE_DEFAULT_ENGINE' in error for error in errors)
        assert any('Invalid CONTENT_PIPELINE_LOG_LEVEL' in error for error in errors)
        assert any('Invalid WHISPER_LOCAL_MODEL' in error for error in errors)
    
    @patch.dict(os.environ, {
        'CONTENT_PIPELINE_DEFAULT_ENGINE': 'openai-whisper'
    }, clear=True)
    def test_validate_environment_setup_with_warnings(self):
        """Test validation with missing API keys for selected engine."""
        warnings, errors = EnvironmentVariables.validate_environment_setup()
        
        assert len(errors) == 0
        assert len(warnings) == 1
        assert 'OPENAI_API_KEY is not set' in warnings[0]
    
    @patch.dict(os.environ, {
        'CONTENT_PIPELINE_DEFAULT_ENGINE': 'aws-transcribe'
    }, clear=True)
    def test_validate_environment_setup_aws_warnings(self):
        """Test validation with missing AWS credentials for selected engine."""
        warnings, errors = EnvironmentVariables.validate_environment_setup()
        
        assert len(errors) == 0
        assert len(warnings) == 2
        assert any('AWS_ACCESS_KEY_ID is not set' in warning for warning in warnings)
        assert any('AWS_SECRET_ACCESS_KEY is not set' in warning for warning in warnings)
    
    @patch.dict(os.environ, {
        'CONTENT_PIPELINE_DEFAULT_ENGINE': 'auto',
        'CONTENT_PIPELINE_LOG_LEVEL': 'info',
        'WHISPER_LOCAL_MODEL': 'base'
    })
    def test_validate_environment_setup_valid(self):
        """Test validation with valid environment variables."""
        warnings, errors = EnvironmentVariables.validate_environment_setup()
        
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_get_setup_instructions(self):
        """Test that setup instructions are provided."""
        instructions = EnvironmentVariables.get_setup_instructions()
        
        assert isinstance(instructions, str)
        assert len(instructions.strip()) > 0
        assert 'export' in instructions
        assert 'CONTENT_PIPELINE_DEFAULT_ENGINE' in instructions
        assert 'OPENAI_API_KEY' in instructions
        assert 'AWS_ACCESS_KEY_ID' in instructions
    
    def test_check_required_for_whisper_local(self):
        """Test that local-whisper requires no environment variables."""
        missing = EnvironmentVariables.check_required_for_engine('local-whisper')
        assert missing == []
    
    @patch.dict(os.environ, {}, clear=True)
    def test_check_required_for_whisper_api_missing(self):
        """Test that openai-whisper requires OPENAI_API_KEY."""
        missing = EnvironmentVariables.check_required_for_engine('openai-whisper')
        assert missing == ['OPENAI_API_KEY']
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_check_required_for_whisper_api_present(self):
        """Test that openai-whisper is satisfied when API key is present."""
        missing = EnvironmentVariables.check_required_for_engine('openai-whisper')
        assert missing == []
    
    @patch.dict(os.environ, {}, clear=True)
    def test_check_required_for_aws_transcribe_missing(self):
        """Test that aws-transcribe requires AWS credentials."""
        missing = EnvironmentVariables.check_required_for_engine('aws-transcribe')
        assert set(missing) == {'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'}
    
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test-key-id',
        'AWS_SECRET_ACCESS_KEY': 'test-secret-key'
    })
    def test_check_required_for_aws_transcribe_present(self):
        """Test that aws-transcribe is satisfied when credentials are present."""
        missing = EnvironmentVariables.check_required_for_engine('aws-transcribe')
        assert missing == []