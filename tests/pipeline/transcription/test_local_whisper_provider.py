"""
Unit Tests: LocalWhisperProvider

Tests for the Local Whisper transcription provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Transcribe method with various inputs
- Estimate cost method (should always return None)
- Validate requirements method
- Error handling for missing files and unsupported formats
- Get supported formats method
- Get engine info method
- Model size info method

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from pathlib import Path

from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.config import WhisperLocalConfig
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
def whisper_config():
    """Create a test Whisper configuration."""
    return WhisperLocalConfig(
        model="base",
        device="cpu",
        compute_type="default",
        timeout=300,
        retry_attempts=3,
        retry_delay=1.0
    )


@pytest.fixture
def whisper_provider(whisper_config):
    """Create a LocalWhisperProvider instance."""
    return LocalWhisperProvider(whisper_config)


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a mock audio file."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_text("fake audio content")
    return str(audio_file)


# ============================================================================
# Test: Initialization
# ============================================================================

def test_initialization_with_valid_config(whisper_config):
    """Test provider initialization with valid configuration."""
    provider = LocalWhisperProvider(whisper_config)
    
    assert provider is not None
    assert provider.config == whisper_config
    assert provider.config.model == "base"
    assert provider.config.device == "cpu"
    assert provider.model is None  # Model not loaded yet
    assert provider._model_loaded is False


def test_initialization_with_invalid_config():
    """Test provider initialization with invalid configuration type."""
    with pytest.raises(ConfigurationError) as exc_info:
        LocalWhisperProvider("not a config object")
    
    assert "Expected WhisperLocalConfig" in str(exc_info.value)


def test_initialization_stores_config(whisper_config):
    """Test that provider stores configuration object."""
    provider = LocalWhisperProvider(whisper_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, WhisperLocalConfig)


# ============================================================================
# Test: Transcribe Method
# ============================================================================

@patch('whisper.load_model')
def test_transcribe_with_valid_file(mock_load_model, whisper_provider, mock_audio_file):
    """Test transcribe method with valid audio file."""
    # Mock Whisper model
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "This is a test transcription.",
        "segments": [{"start": 0.0, "end": 2.0, "text": "This is a test transcription."}],
        "language": "en"
    }
    mock_load_model.return_value = mock_model
    
    # Transcribe
    result = whisper_provider.transcribe(mock_audio_file, language="en")
    
    # Verify result
    assert result["text"] == "This is a test transcription."
    assert "segments" in result
    assert result["language"] == "en"
    
    # Verify model was loaded
    mock_load_model.assert_called_once()
    
    # Verify transcribe was called
    mock_model.transcribe.assert_called_once_with(mock_audio_file, language="en")


@patch('whisper.load_model')
def test_transcribe_without_language(mock_load_model, whisper_provider, mock_audio_file):
    """Test transcribe method without language hint."""
    # Mock Whisper model
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "Test",
        "segments": [],
        "language": "en"
    }
    mock_load_model.return_value = mock_model
    
    # Transcribe without language
    result = whisper_provider.transcribe(mock_audio_file)
    
    # Verify transcribe was called without language
    mock_model.transcribe.assert_called_once_with(mock_audio_file, language=None)


def test_transcribe_with_nonexistent_file(whisper_provider):
    """Test transcribe method with nonexistent file."""
    with pytest.raises(AudioFileError) as exc_info:
        whisper_provider.transcribe("/nonexistent/file.mp3")
    
    assert "not found" in str(exc_info.value)


def test_transcribe_with_unsupported_format(whisper_provider, tmp_path):
    """Test transcribe method with unsupported audio format."""
    # Create file with unsupported extension
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("fake content")
    
    with pytest.raises(AudioFileError) as exc_info:
        whisper_provider.transcribe(str(unsupported_file))
    
    assert "Unsupported audio format" in str(exc_info.value)


@patch('whisper.load_model')
def test_transcribe_model_failure(mock_load_model, whisper_provider, mock_audio_file):
    """Test transcribe method when model transcription fails."""
    # Mock Whisper model that raises exception
    mock_model = Mock()
    mock_model.transcribe.side_effect = Exception("Transcription failed")
    mock_load_model.return_value = mock_model
    
    with pytest.raises(ProviderError) as exc_info:
        whisper_provider.transcribe(mock_audio_file)
    
    assert "transcription failed" in str(exc_info.value).lower()


# ============================================================================
# Test: Model Loading
# ============================================================================

@patch('whisper.load_model')
def test_load_model_with_cpu_device(mock_load_model, whisper_config):
    """Test model loading with CPU device."""
    whisper_config.device = "cpu"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock model
    mock_model = Mock()
    mock_load_model.return_value = mock_model
    
    # Load model
    provider._load_model()
    
    # Verify model was loaded with correct device
    mock_load_model.assert_called_once_with("base", device="cpu")
    assert provider._model_loaded is True


@patch('whisper.load_model')
def test_load_model_with_auto_device(mock_load_model, whisper_config):
    """Test model loading with auto device selection."""
    whisper_config.device = "auto"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock model
    mock_model = Mock()
    mock_load_model.return_value = mock_model
    
    # Load model
    provider._load_model()
    
    # Verify model was loaded without device parameter (auto selection)
    mock_load_model.assert_called_once_with("base")
    assert provider._model_loaded is True


@patch('whisper.load_model')
def test_load_model_failure(mock_load_model, whisper_provider):
    """Test model loading failure."""
    mock_load_model.side_effect = Exception("Failed to load model")
    
    with pytest.raises(ProviderError) as exc_info:
        whisper_provider._load_model()
    
    assert "Failed to load" in str(exc_info.value)


@patch('whisper.load_model')
def test_load_model_only_once(mock_load_model, whisper_provider, mock_audio_file):
    """Test that model is only loaded once."""
    # Mock model
    mock_model = Mock()
    mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
    mock_load_model.return_value = mock_model
    
    # Transcribe multiple times
    whisper_provider.transcribe(mock_audio_file)
    whisper_provider.transcribe(mock_audio_file)
    
    # Verify model was loaded only once
    mock_load_model.assert_called_once()


# ============================================================================
# Test: Estimate Cost
# ============================================================================

def test_estimate_cost_returns_none(whisper_provider):
    """Test estimate_cost always returns None for local models."""
    cost = whisper_provider.estimate_cost(300.0)  # 5 minutes
    
    assert cost is None


def test_estimate_cost_with_different_durations(whisper_provider):
    """Test estimate_cost returns None regardless of duration."""
    assert whisper_provider.estimate_cost(0.0) is None
    assert whisper_provider.estimate_cost(60.0) is None
    assert whisper_provider.estimate_cost(3600.0) is None


# ============================================================================
# Test: Validate Requirements
# ============================================================================

@patch('whisper.available_models')
@patch('whisper.load_model')
def test_validate_requirements_success(mock_load_model, mock_available_models, whisper_provider):
    """Test validate_requirements when all requirements are met."""
    # Mock available models
    mock_available_models.return_value = ['tiny', 'base', 'small', 'medium', 'large']
    
    # Mock model loading
    mock_model = Mock()
    mock_load_model.return_value = mock_model
    
    errors = whisper_provider.validate_requirements()
    
    assert len(errors) == 0


@patch('whisper.available_models')
def test_validate_requirements_model_not_available(mock_available_models, whisper_config):
    """Test validate_requirements when model is not available."""
    whisper_config.model = "nonexistent"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock available models
    mock_available_models.return_value = ['tiny', 'base', 'small']
    
    errors = provider.validate_requirements()
    
    assert len(errors) > 0
    assert any("not available" in error for error in errors)


@patch('whisper.available_models')
@patch('torch.cuda.is_available')
def test_validate_requirements_cuda_not_available(mock_cuda_available, mock_available_models, whisper_config):
    """Test validate_requirements when CUDA is requested but not available."""
    whisper_config.device = "cuda"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock available models
    mock_available_models.return_value = ['base']
    
    # Mock CUDA not available
    mock_cuda_available.return_value = False
    
    errors = provider.validate_requirements()
    
    assert len(errors) > 0
    assert any("CUDA" in error for error in errors)


# ============================================================================
# Test: Get Supported Formats
# ============================================================================

def test_get_supported_formats(whisper_provider):
    """Test get_supported_formats returns correct formats."""
    formats = whisper_provider.get_supported_formats()
    
    assert isinstance(formats, list)
    assert len(formats) > 0
    assert 'mp3' in formats
    assert 'wav' in formats
    assert 'flac' in formats
    assert 'm4a' in formats


def test_get_supported_formats_returns_copy(whisper_provider):
    """Test get_supported_formats returns a copy, not the original list."""
    formats1 = whisper_provider.get_supported_formats()
    formats2 = whisper_provider.get_supported_formats()
    
    # Modify one list
    formats1.append('xyz')
    
    # Verify the other list is not affected
    assert 'xyz' not in formats2


# ============================================================================
# Test: Get Engine Info
# ============================================================================

def test_get_engine_info(whisper_provider):
    """Test get_engine_info returns correct information."""
    name, version = whisper_provider.get_engine_info()
    
    assert name == "local-whisper"
    assert version == "base"


def test_get_engine_info_with_different_model(whisper_config):
    """Test get_engine_info with different model."""
    whisper_config.model = "large"
    provider = LocalWhisperProvider(whisper_config)
    
    name, version = provider.get_engine_info()
    
    assert name == "local-whisper"
    assert version == "large"


# ============================================================================
# Test: Get Model Size Info
# ============================================================================

def test_get_model_size_info(whisper_provider):
    """Test get_model_size_info returns correct information."""
    info = whisper_provider.get_model_size_info()
    
    assert info['model'] == 'base'
    assert info['device'] == 'cpu'
    assert 'info' in info
    assert 'params' in info['info']
    assert 'vram' in info['info']
    assert 'speed' in info['info']


def test_get_model_size_info_for_different_models(whisper_config):
    """Test get_model_size_info for different model sizes."""
    test_models = ['tiny', 'base', 'small', 'medium', 'large']
    
    for model_name in test_models:
        whisper_config.model = model_name
        provider = LocalWhisperProvider(whisper_config)
        
        info = provider.get_model_size_info()
        
        assert info['model'] == model_name
        assert info['info']['params'] != 'unknown'


# ============================================================================
# Test: Configuration Usage
# ============================================================================

@patch('whisper.load_model')
def test_uses_config_model(mock_load_model, whisper_config, mock_audio_file):
    """Test that provider uses model from config."""
    whisper_config.model = "small"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock model
    mock_model = Mock()
    mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
    mock_load_model.return_value = mock_model
    
    provider.transcribe(mock_audio_file)
    
    # Verify correct model was loaded
    mock_load_model.assert_called_once()
    assert mock_load_model.call_args[0][0] == "small"


@patch('whisper.load_model')
def test_uses_config_device(mock_load_model, whisper_config, mock_audio_file):
    """Test that provider uses device from config."""
    whisper_config.device = "cuda"
    provider = LocalWhisperProvider(whisper_config)
    
    # Mock model
    mock_model = Mock()
    mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
    mock_load_model.return_value = mock_model
    
    provider.transcribe(mock_audio_file)
    
    # Verify correct device was used
    mock_load_model.assert_called_once()
    assert mock_load_model.call_args[1]['device'] == "cuda"
