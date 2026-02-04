"""
Unit Tests: CloudOpenAIWhisperProvider

Tests for the Cloud OpenAI Whisper transcription provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Transcribe method with various inputs
- Estimate cost method with accurate pricing
- Validate requirements method
- Error handling for missing files, unsupported formats, and file size limits
- Get supported formats method
- Get engine info method
- Model info method

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from pathlib import Path

from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.config import WhisperAPIConfig
from pipeline.transcription.errors import (
    AudioFileError,
    ProviderError,
    ProviderNotAvailableError,
    ConfigurationError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def whisper_api_config():
    """Create a test Whisper API configuration."""
    return WhisperAPIConfig(
        api_key="sk-test-key-12345",
        model="whisper-1",
        temperature=0.0,
        response_format="json",
        timeout=60,
        retry_attempts=3,
        retry_delay=2.0
    )


@pytest.fixture
def whisper_api_provider(whisper_api_config):
    """Create a CloudOpenAIWhisperProvider instance."""
    return CloudOpenAIWhisperProvider(whisper_api_config)


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a mock audio file."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_text("fake audio content")
    return str(audio_file)


# ============================================================================
# Test: Initialization
# ============================================================================

def test_initialization_with_valid_config(whisper_api_config):
    """Test provider initialization with valid configuration."""
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    assert provider is not None
    assert provider.config == whisper_api_config
    assert provider.config.api_key == "sk-test-key-12345"
    assert provider.config.model == "whisper-1"
    assert provider.client is None  # Client not initialized yet


def test_initialization_with_invalid_config():
    """Test provider initialization with invalid configuration type."""
    with pytest.raises(ConfigurationError) as exc_info:
        CloudOpenAIWhisperProvider("not a config object")
    
    assert "Expected WhisperAPIConfig" in str(exc_info.value)


def test_initialization_stores_config(whisper_api_config):
    """Test that provider stores configuration object."""
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, WhisperAPIConfig)


# ============================================================================
# Test: Transcribe Method
# ============================================================================

@patch('builtins.open', create=True)
def test_transcribe_with_valid_file(mock_open, whisper_api_provider, mock_audio_file):
    """Test transcribe method with valid audio file."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "This is a test transcription."
    # Mock getattr for segments and language
    mock_response.segments = []
    mock_response.language = "en"
    mock_client.audio.transcriptions.create.return_value = mock_response
    whisper_api_provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Transcribe
    result = whisper_api_provider.transcribe(mock_audio_file, language="en")
    
    # Verify result
    assert result["text"] == "This is a test transcription."
    assert "segments" in result
    assert result["language"] == "en"
    
    # Verify API was called
    mock_client.audio.transcriptions.create.assert_called_once()


@patch('builtins.open', create=True)
def test_transcribe_with_verbose_json_format(mock_open, whisper_api_config, mock_audio_file):
    """Test transcribe method with verbose_json response format."""
    whisper_api_config.response_format = "verbose_json"
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "Test transcription"
    mock_response.segments = [{"start": 0.0, "end": 2.0, "text": "Test transcription"}]
    mock_response.language = "en"
    mock_response.duration = 2.0
    mock_response.words = [{"word": "Test", "start": 0.0, "end": 1.0}]
    mock_client.audio.transcriptions.create.return_value = mock_response
    provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Transcribe
    result = provider.transcribe(mock_audio_file)
    
    # Verify result has verbose fields
    assert result["text"] == "Test transcription"
    assert result["segments"] == mock_response.segments
    assert result["language"] == "en"
    assert result["duration"] == 2.0
    assert result["words"] == mock_response.words


def test_transcribe_with_nonexistent_file(whisper_api_provider):
    """Test transcribe method with nonexistent file."""
    with pytest.raises(AudioFileError) as exc_info:
        whisper_api_provider.transcribe("/nonexistent/file.mp3")
    
    assert "not found" in str(exc_info.value)


def test_transcribe_with_unsupported_format(whisper_api_provider, tmp_path):
    """Test transcribe method with unsupported audio format."""
    # Create file with unsupported extension
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("fake content")
    
    with pytest.raises(AudioFileError) as exc_info:
        whisper_api_provider.transcribe(str(unsupported_file))
    
    assert "Unsupported audio format" in str(exc_info.value)


def test_transcribe_with_file_too_large(whisper_api_provider, tmp_path):
    """Test transcribe method with file exceeding size limit."""
    # Create a file that reports as too large
    large_file = tmp_path / "large_audio.mp3"
    large_file.write_text("x" * 100)  # Small file, but we'll mock the size
    
    with patch('os.path.getsize', return_value=26 * 1024 * 1024):  # 26 MB
        with pytest.raises(AudioFileError) as exc_info:
            whisper_api_provider.transcribe(str(large_file))
        
        assert "too large" in str(exc_info.value).lower()


@patch('builtins.open', create=True)
def test_transcribe_api_failure(mock_open, whisper_api_provider, mock_audio_file):
    """Test transcribe method when API call fails."""
    # Mock OpenAI client that raises exception
    mock_client = Mock()
    mock_client.audio.transcriptions.create.side_effect = Exception("API error")
    whisper_api_provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    with pytest.raises(ProviderError) as exc_info:
        whisper_api_provider.transcribe(mock_audio_file)
    
    assert "transcription failed" in str(exc_info.value).lower()


# ============================================================================
# Test: Client Initialization
# ============================================================================

def test_ensure_client_initialized(whisper_api_provider):
    """Test client initialization."""
    # We can't easily test this without importing openai, so we'll just verify
    # that the method exists and can be called
    with patch('openai.OpenAI') as mock_openai_class:
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        whisper_api_provider._ensure_client_initialized()
        
        assert whisper_api_provider.client is not None
        mock_openai_class.assert_called_once_with(api_key="sk-test-key-12345")


def test_ensure_client_initialized_without_openai_package(whisper_api_provider):
    """Test client initialization when OpenAI package is not installed."""
    # Mock the import to fail
    with patch.dict('sys.modules', {'openai': None}):
        # Force re-import by clearing the client
        whisper_api_provider.client = None
        
        with pytest.raises(ProviderNotAvailableError) as exc_info:
            whisper_api_provider._ensure_client_initialized()
        
        assert "not installed" in str(exc_info.value)


# ============================================================================
# Test: Estimate Cost
# ============================================================================

def test_estimate_cost_calculates_correctly(whisper_api_provider):
    """Test estimate_cost calculates cost based on duration."""
    # 5 minutes = 300 seconds
    cost = whisper_api_provider.estimate_cost(300.0)
    
    # Cost should be 5 * $0.006 = $0.03
    assert cost == 0.03


def test_estimate_cost_with_zero_duration(whisper_api_provider):
    """Test estimate_cost with zero duration."""
    cost = whisper_api_provider.estimate_cost(0.0)
    
    assert cost == 0.0


def test_estimate_cost_with_negative_duration(whisper_api_provider):
    """Test estimate_cost with negative duration."""
    cost = whisper_api_provider.estimate_cost(-10.0)
    
    assert cost == 0.0


def test_estimate_cost_rounds_correctly(whisper_api_provider):
    """Test estimate_cost rounds to 4 decimal places."""
    # 123 seconds = 2.05 minutes = $0.0123
    cost = whisper_api_provider.estimate_cost(123.0)
    
    assert isinstance(cost, float)
    assert len(str(cost).split('.')[-1]) <= 4  # At most 4 decimal places


# ============================================================================
# Test: Validate Requirements
# ============================================================================

def test_validate_requirements_with_valid_key(whisper_api_provider):
    """Test validate_requirements with valid API key."""
    with patch('openai.OpenAI'):
        errors = whisper_api_provider.validate_requirements()
        
        # Should have no errors (we don't test actual API connectivity)
        assert len(errors) == 0


def test_validate_requirements_without_api_key():
    """Test validate_requirements without API key."""
    config = WhisperAPIConfig(api_key="")
    provider = CloudOpenAIWhisperProvider(config)
    
    errors = provider.validate_requirements()
    
    assert len(errors) > 0
    assert any("API key not found" in error for error in errors)


def test_validate_requirements_with_invalid_key_format(whisper_api_config):
    """Test validate_requirements with invalid API key format."""
    whisper_api_config.api_key = "invalid-key-format"
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    errors = provider.validate_requirements()
    
    assert len(errors) > 0
    assert any("Invalid" in error and "API key" in error for error in errors)


def test_validate_requirements_without_openai_package(whisper_api_provider):
    """Test validate_requirements when OpenAI package is not installed."""
    with patch.dict('sys.modules', {'openai': None}):
        errors = whisper_api_provider.validate_requirements()
        
        assert len(errors) > 0
        assert any("not installed" in error for error in errors)


# ============================================================================
# Test: Get Supported Formats
# ============================================================================

def test_get_supported_formats(whisper_api_provider):
    """Test get_supported_formats returns correct formats."""
    formats = whisper_api_provider.get_supported_formats()
    
    assert isinstance(formats, list)
    assert len(formats) > 0
    assert 'mp3' in formats
    assert 'mp4' in formats
    assert 'wav' in formats
    assert 'webm' in formats


def test_get_supported_formats_returns_copy(whisper_api_provider):
    """Test get_supported_formats returns a copy, not the original list."""
    formats1 = whisper_api_provider.get_supported_formats()
    formats2 = whisper_api_provider.get_supported_formats()
    
    # Modify one list
    formats1.append('xyz')
    
    # Verify the other list is not affected
    assert 'xyz' not in formats2


# ============================================================================
# Test: Get Engine Info
# ============================================================================

def test_get_engine_info(whisper_api_provider):
    """Test get_engine_info returns correct information."""
    name, version = whisper_api_provider.get_engine_info()
    
    assert name == "cloud-openai-whisper"
    assert version == "whisper-1"


def test_get_engine_info_with_different_model(whisper_api_config):
    """Test get_engine_info with different model."""
    whisper_api_config.model = "whisper-2"  # Hypothetical future model
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    name, version = provider.get_engine_info()
    
    assert name == "cloud-openai-whisper"
    assert version == "whisper-2"


# ============================================================================
# Test: Get Model Info
# ============================================================================

def test_get_model_info(whisper_api_provider):
    """Test get_model_info returns correct information."""
    info = whisper_api_provider.get_model_info()
    
    assert info['model'] == 'whisper-1'
    assert info['temperature'] == 0.0
    assert info['response_format'] == 'json'
    assert info['max_file_size_mb'] == 25.0
    assert info['cost_per_minute_usd'] == 0.006
    assert info['api_key_configured'] is True


def test_get_model_info_without_api_key():
    """Test get_model_info when API key is not configured."""
    config = WhisperAPIConfig(api_key=None)
    provider = CloudOpenAIWhisperProvider(config)
    
    info = provider.get_model_info()
    
    assert info['api_key_configured'] is False


# ============================================================================
# Test: Get File Size Limit
# ============================================================================

def test_get_file_size_limit(whisper_api_provider):
    """Test get_file_size_limit returns correct limit."""
    limit = whisper_api_provider.get_file_size_limit()
    
    assert limit == 25 * 1024 * 1024  # 25 MB in bytes


# ============================================================================
# Test: Configuration Usage
# ============================================================================

@patch('builtins.open', create=True)
def test_uses_config_model(mock_open, whisper_api_config, mock_audio_file):
    """Test that provider uses model from config."""
    whisper_api_config.model = "whisper-1"
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "Test"
    mock_client.audio.transcriptions.create.return_value = mock_response
    provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    provider.transcribe(mock_audio_file)
    
    # Verify correct model was used
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs['model'] == "whisper-1"


@patch('builtins.open', create=True)
def test_uses_config_temperature(mock_open, whisper_api_config, mock_audio_file):
    """Test that provider uses temperature from config."""
    whisper_api_config.temperature = 0.5
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "Test"
    mock_client.audio.transcriptions.create.return_value = mock_response
    provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    provider.transcribe(mock_audio_file)
    
    # Verify correct temperature was used
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs['temperature'] == 0.5


@patch('builtins.open', create=True)
def test_uses_config_response_format(mock_open, whisper_api_config, mock_audio_file):
    """Test that provider uses response_format from config."""
    whisper_api_config.response_format = "srt"
    provider = CloudOpenAIWhisperProvider(whisper_api_config)
    
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="1\n00:00:00,000 --> 00:00:02,000\nTest")
    mock_client.audio.transcriptions.create.return_value = mock_response
    provider.client = mock_client
    
    # Mock file open
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    result = provider.transcribe(mock_audio_file)
    
    # Verify correct response format was used
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs['response_format'] == "srt"


# ============================================================================
# Test: Pricing Configuration Override (Task 30.1)
# ============================================================================

def test_cost_per_minute_override_via_config():
    """Test that cost_per_minute_usd can be overridden via config."""
    config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=0.010  # Override default $0.006
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    # Verify config stores the override
    assert provider.config.cost_per_minute_usd == 0.010


def test_estimate_cost_uses_config_override():
    """Test that estimate_cost uses cost_per_minute_usd from config."""
    config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=0.010  # Override default $0.006
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    # 5 minutes = 300 seconds
    cost = provider.estimate_cost(300.0)
    
    # Cost should be 5 * $0.010 = $0.05 (not default $0.03)
    assert cost == 0.05


def test_get_model_info_returns_config_cost():
    """Test that get_model_info returns cost_per_minute_usd from config."""
    config = WhisperAPIConfig(
        api_key="sk-test-key",
        cost_per_minute_usd=0.010  # Override default $0.006
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    info = provider.get_model_info()
    
    assert info['cost_per_minute_usd'] == 0.010


def test_default_cost_per_minute_when_not_overridden():
    """Test that default cost_per_minute_usd is used when not overridden."""
    config = WhisperAPIConfig(
        api_key="sk-test-key"
        # cost_per_minute_usd not specified, should use default
    )
    provider = CloudOpenAIWhisperProvider(config)
    
    # Verify default value is used
    assert provider.config.cost_per_minute_usd == 0.006
    
    # Verify estimate_cost uses default
    cost = provider.estimate_cost(60.0)  # 1 minute
    assert cost == 0.006


@patch.dict(os.environ, {'WHISPER_API_COST_PER_MINUTE': '0.012'})
def test_cost_per_minute_override_via_environment(tmp_path):
    """Test that cost_per_minute_usd can be overridden via environment variable."""
    from pipeline.transcription.config import TranscriptionConfig
    
    # Create a minimal config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
whisper_api:
  api_key: sk-test-key
""")
    
    # Load config with environment variable override
    full_config = TranscriptionConfig.load_from_yaml(str(config_file))
    config = full_config.whisper_api
    
    # Verify environment variable was applied
    assert config.cost_per_minute_usd == 0.012
    
    # Verify provider uses the override
    provider = CloudOpenAIWhisperProvider(config)
    cost = provider.estimate_cost(60.0)  # 1 minute
    assert cost == 0.012
