"""
Unit Tests for Transcription Configuration

Tests configuration loading, environment variable substitution,
and default value fallback for transcription providers.

**Validates: Requirements 2.5, 7.2, 7.4, 10.3**
"""

import pytest
import os
import tempfile
from pathlib import Path

from pipeline.transcription.config import (
    WhisperLocalConfig,
    WhisperAPIConfig,
    AWSTranscribeConfig,
    TranscriptionConfig
)


class TestWhisperLocalConfig:
    """Test WhisperLocalConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WhisperLocalConfig()
        assert config.model == "base"
        assert config.device == "auto"
        assert config.compute_type == "default"
        assert config.timeout == 300
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = WhisperLocalConfig(
            model="large",
            device="cuda",
            compute_type="int8",
            timeout=600,
            retry_attempts=5,
            retry_delay=2.0
        )
        assert config.model == "large"
        assert config.device == "cuda"
        assert config.compute_type == "int8"
        assert config.timeout == 600
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0


class TestWhisperAPIConfig:
    """Test WhisperAPIConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WhisperAPIConfig()
        assert config.api_key is None
        assert config.model == "whisper-1"
        assert config.temperature == 0.0
        assert config.response_format == "json"
        assert config.timeout == 60
        assert config.retry_attempts == 3
        assert config.retry_delay == 2.0
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = WhisperAPIConfig(
            api_key="sk-test123",
            model="whisper-1",
            temperature=0.5,
            response_format="verbose_json",
            timeout=120,
            retry_attempts=5,
            retry_delay=3.0
        )
        assert config.api_key == "sk-test123"
        assert config.model == "whisper-1"
        assert config.temperature == 0.5
        assert config.response_format == "verbose_json"
        assert config.timeout == 120
        assert config.retry_attempts == 5
        assert config.retry_delay == 3.0


class TestAWSTranscribeConfig:
    """Test AWSTranscribeConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AWSTranscribeConfig()
        assert config.access_key_id is None
        assert config.secret_access_key is None
        assert config.region == "us-east-1"
        assert config.language_code == "en-US"
        assert config.s3_bucket is None
        assert config.timeout == 600
        assert config.retry_attempts == 3
        assert config.retry_delay == 2.0
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = AWSTranscribeConfig(
            access_key_id="AKIATEST",
            secret_access_key="secret123",
            region="us-west-2",
            language_code="es-ES",
            s3_bucket="my-bucket",
            timeout=900,
            retry_attempts=5,
            retry_delay=3.0
        )
        assert config.access_key_id == "AKIATEST"
        assert config.secret_access_key == "secret123"
        assert config.region == "us-west-2"
        assert config.language_code == "es-ES"
        assert config.s3_bucket == "my-bucket"
        assert config.timeout == 900
        assert config.retry_attempts == 5
        assert config.retry_delay == 3.0


class TestTranscriptionConfig:
    """Test TranscriptionConfig main class."""
    
    def test_default_initialization(self):
        """Test that TranscriptionConfig initializes with defaults."""
        config = TranscriptionConfig()
        assert isinstance(config.whisper_local, WhisperLocalConfig)
        assert isinstance(config.whisper_api, WhisperAPIConfig)
        assert isinstance(config.aws_transcribe, AWSTranscribeConfig)
    
    def test_load_from_yaml_basic(self):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_content = """
whisper_local:
  model: medium
  device: cpu
  timeout: 400

whisper_api:
  api_key: sk-test-key
  model: whisper-1
  temperature: 0.3

aws_transcribe:
  region: eu-west-1
  language_code: fr-FR
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            
            # Check whisper_local
            assert config.whisper_local.model == "medium"
            assert config.whisper_local.device == "cpu"
            assert config.whisper_local.timeout == 400
            
            # Check whisper_api
            assert config.whisper_api.api_key == "sk-test-key"
            assert config.whisper_api.model == "whisper-1"
            assert config.whisper_api.temperature == 0.3
            
            # Check aws_transcribe
            assert config.aws_transcribe.region == "eu-west-1"
            assert config.aws_transcribe.language_code == "fr-FR"
        finally:
            os.unlink(config_path)
    
    def test_load_from_yaml_with_env_substitution(self):
        """Test environment variable substitution in YAML."""
        # Set environment variables
        os.environ['TEST_WHISPER_MODEL'] = 'large'
        os.environ['TEST_API_KEY'] = 'sk-env-key'
        os.environ['TEST_AWS_REGION'] = 'ap-southeast-1'
        
        config_content = """
whisper_local:
  model: ${TEST_WHISPER_MODEL:-base}
  device: auto

whisper_api:
  api_key: ${TEST_API_KEY:-}
  model: whisper-1

aws_transcribe:
  region: ${TEST_AWS_REGION:-us-east-1}
  language_code: en-US
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            
            assert config.whisper_local.model == "large"
            assert config.whisper_api.api_key == "sk-env-key"
            assert config.aws_transcribe.region == "ap-southeast-1"
        finally:
            os.unlink(config_path)
            # Clean up environment variables
            del os.environ['TEST_WHISPER_MODEL']
            del os.environ['TEST_API_KEY']
            del os.environ['TEST_AWS_REGION']
    
    def test_load_from_yaml_with_default_fallback(self):
        """Test that default values are used when config and env are not set."""
        config_content = """
whisper_local:
  model: ${NONEXISTENT_VAR:-small}

whisper_api:
  api_key: ${NONEXISTENT_API_KEY:-}

aws_transcribe:
  region: ${NONEXISTENT_REGION:-}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            
            # Should use default from ${VAR:-default} syntax
            assert config.whisper_local.model == "small"
            
            # Should use None when no default in substitution
            assert config.whisper_api.api_key is None
            
            # Should use system default when config value is None
            assert config.aws_transcribe.region == "us-east-1"
        finally:
            os.unlink(config_path)
    
    def test_load_from_yaml_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            TranscriptionConfig.load_from_yaml("/nonexistent/config.yaml")
    
    def test_load_from_yaml_invalid_yaml(self):
        """Test that ValueError is raised for invalid YAML."""
        config_content = """
whisper_local:
  model: medium
  invalid yaml here: [unclosed bracket
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                TranscriptionConfig.load_from_yaml(config_path)
        finally:
            os.unlink(config_path)
    
    def test_resolve_value_precedence(self):
        """Test configuration value resolution precedence."""
        # Test 1: Config value takes precedence
        result = TranscriptionConfig._resolve_value("config_value", "ENV_VAR", "default")
        assert result == "config_value"
        
        # Test 2: Environment variable used when config is None
        os.environ['TEST_ENV_VAR'] = 'env_value'
        result = TranscriptionConfig._resolve_value(None, "TEST_ENV_VAR", "default")
        assert result == "env_value"
        del os.environ['TEST_ENV_VAR']
        
        # Test 3: Default used when both config and env are None
        result = TranscriptionConfig._resolve_value(None, "NONEXISTENT_VAR", "default")
        assert result == "default"
        
        # Test 4: Empty string treated as None
        result = TranscriptionConfig._resolve_value("", "NONEXISTENT_VAR", "default")
        assert result == "default"
    
    def test_resolve_value_with_env_substitution_syntax(self):
        """Test environment variable substitution syntax."""
        # Test ${VAR:-default} syntax
        os.environ['TEST_VAR'] = 'test_value'
        result = TranscriptionConfig._resolve_value("${TEST_VAR:-default}", "UNUSED", "unused")
        assert result == "test_value"
        del os.environ['TEST_VAR']
        
        # Test ${VAR:-default} with missing var (should use default from syntax)
        result = TranscriptionConfig._resolve_value("${MISSING_VAR:-fallback}", "UNUSED", "unused")
        assert result == "fallback"
        
        # Test ${VAR} without default
        result = TranscriptionConfig._resolve_value("${MISSING_VAR}", "UNUSED", "system_default")
        # Should resolve to empty string, then fall back to system default
        assert result == "system_default"
    
    def test_load_from_yaml_empty_sections(self):
        """Test loading config with empty sections."""
        config_content = """
whisper_local:
whisper_api:
aws_transcribe:
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            
            # Should use all defaults
            assert config.whisper_local.model == "base"
            assert config.whisper_api.model == "whisper-1"
            assert config.aws_transcribe.region == "us-east-1"
        finally:
            os.unlink(config_path)
    
    def test_load_from_yaml_missing_sections(self):
        """Test loading config with missing sections."""
        config_content = """
# Only whisper_local section
whisper_local:
  model: tiny
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            
            # whisper_local should have custom value
            assert config.whisper_local.model == "tiny"
            
            # Other sections should use defaults
            assert config.whisper_api.model == "whisper-1"
            assert config.aws_transcribe.region == "us-east-1"
        finally:
            os.unlink(config_path)
    
    @pytest.mark.parametrize("model,expected", [
        ("tiny", "tiny"),
        ("base", "base"),
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
    ])
    def test_whisper_model_values(self, model, expected):
        """Test different Whisper model values."""
        config_content = f"""
whisper_local:
  model: {model}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            assert config.whisper_local.model == expected
        finally:
            os.unlink(config_path)
    
    @pytest.mark.parametrize("device,expected", [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("auto", "auto"),
    ])
    def test_whisper_device_values(self, device, expected):
        """Test different device values."""
        config_content = f"""
whisper_local:
  device: {device}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config = TranscriptionConfig.load_from_yaml(config_path)
            assert config.whisper_local.device == expected
        finally:
            os.unlink(config_path)
