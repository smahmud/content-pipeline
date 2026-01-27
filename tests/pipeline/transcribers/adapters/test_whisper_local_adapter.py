"""
File: test_whisper_local_adapter.py

Unit tests for WhisperLocalAdapter behavior and error handling.

Covers:
- Local Whisper transcription functionality
- Device selection and model loading
- Enhanced error handling and validation
- Protocol compliance and configuration
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipeline.transcribers.adapters.whisper_local import WhisperLocalAdapter
from pipeline.transcribers.adapters.base import TranscriberAdapter


class TestWhisperLocalAdapter:
    """Test the WhisperLocalAdapter class."""

    def test_adapter_implements_enhanced_protocol(self):
        """Test that WhisperLocalAdapter implements all enhanced protocol methods."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Check that all protocol methods exist
        assert hasattr(adapter, 'transcribe')
        assert hasattr(adapter, 'get_engine_info')
        assert hasattr(adapter, 'validate_requirements')
        assert hasattr(adapter, 'get_supported_formats')
        assert hasattr(adapter, 'estimate_cost')
        
        # Check that methods are callable
        assert callable(adapter.transcribe)
        assert callable(adapter.get_engine_info)
        assert callable(adapter.validate_requirements)
        assert callable(adapter.get_supported_formats)
        assert callable(adapter.estimate_cost)

    def test_adapter_complies_with_protocol(self):
        """Test that WhisperLocalAdapter is recognized as a TranscriberAdapter."""
        adapter: TranscriberAdapter = WhisperLocalAdapter()
        engine, version = adapter.get_engine_info()
        assert isinstance(engine, str)
        assert isinstance(version, str)
        assert engine == "whisper-local"

    def test_initialization_with_default_parameters(self):
        """Test adapter initialization with default parameters."""
        adapter = WhisperLocalAdapter()
        
        assert adapter.model_name == "base"
        assert adapter.device == "cpu"
        assert adapter.model is None
        assert adapter._model_loaded is False

    def test_initialization_with_custom_parameters(self):
        """Test adapter initialization with custom parameters."""
        adapter = WhisperLocalAdapter(model_name="large", device="cpu")
        
        assert adapter.model_name == "large"
        assert adapter.device == "cpu"
        assert adapter.model is None
        assert adapter._model_loaded is False

    def test_get_supported_formats_returns_list(self):
        """Test that get_supported_formats returns a list of strings."""
        adapter = WhisperLocalAdapter()
        formats = adapter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        
        # Check for common audio formats
        expected_formats = ['mp3', 'wav', 'flac']
        for fmt in expected_formats:
            assert fmt in formats

    def test_estimate_cost_returns_none(self):
        """Test that local Whisper returns None for cost estimation."""
        adapter = WhisperLocalAdapter()
        cost = adapter.estimate_cost(60.0)  # 1 minute of audio
        
        assert cost is None  # Local Whisper is free

    def test_get_engine_info_returns_correct_info(self):
        """Test that get_engine_info returns correct engine information."""
        adapter = WhisperLocalAdapter(model_name="large")
        engine_info = adapter.get_engine_info()
        
        assert isinstance(engine_info, tuple)
        assert len(engine_info) == 2
        assert engine_info[0] == "whisper-local"
        assert engine_info[1] == "large"

    def test_get_model_size_info(self):
        """Test that get_model_size_info returns model information."""
        adapter = WhisperLocalAdapter(model_name="base", device="cpu")
        info = adapter.get_model_size_info()
        
        assert isinstance(info, dict)
        assert 'model' in info
        assert 'device' in info
        assert 'info' in info
        
        assert info['model'] == 'base'
        assert info['device'] == 'cpu'
        assert isinstance(info['info'], dict)

    @patch('whisper.load_model')
    @patch('whisper.available_models')
    def test_validate_requirements_success(self, mock_available_models, mock_load_model):
        """Test validate_requirements when all requirements are met."""
        # Mock successful model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_available_models.return_value = ['tiny', 'base', 'small', 'medium', 'large']
        
        adapter = WhisperLocalAdapter(model_name="base")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) == 0  # No errors when everything is working

    @patch('whisper.load_model')
    def test_validate_requirements_model_load_failure(self, mock_load_model):
        """Test validate_requirements when model loading fails."""
        # Mock model loading failure
        mock_load_model.side_effect = Exception("Model download failed")
        
        adapter = WhisperLocalAdapter(model_name="base")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("Failed to load local Whisper model" in error for error in errors)

    @patch('whisper.available_models')
    @patch('whisper.load_model')
    def test_validate_requirements_invalid_model(self, mock_load_model, mock_available_models):
        """Test validate_requirements with invalid model name."""
        mock_available_models.return_value = ['tiny', 'base', 'small']
        mock_load_model.side_effect = Exception("Model not found")
        
        adapter = WhisperLocalAdapter(model_name="invalid_model")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("Local Whisper model 'invalid_model' not available" in error for error in errors)

    def test_validate_requirements_missing_whisper_package(self):
        """Test validate_requirements when whisper package is not installed."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Mock import error for whisper package
        with patch('builtins.__import__', side_effect=ImportError("No module named 'whisper'")):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("OpenAI Whisper package not installed" in error for error in errors)

    @patch('torch.cuda.is_available')
    def test_validate_requirements_cuda_not_available(self, mock_cuda_available):
        """Test validate_requirements when CUDA is requested but not available."""
        mock_cuda_available.return_value = False
        
        adapter = WhisperLocalAdapter(model_name="base", device="cuda")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("CUDA device requested but not available" in error for error in errors)

    def test_transcribe_validates_file_existence(self):
        """Test that transcribe method validates file existence."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            adapter.transcribe("nonexistent_file.mp3")

    def test_transcribe_validates_file_format(self):
        """Test that transcribe method validates file format."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Create a temporary file with unsupported extension
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported audio format"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    @patch('whisper.load_model')
    def test_transcribe_with_unloaded_model(self, mock_load_model):
        """Test transcribe behavior when model loading fails."""
        mock_load_model.side_effect = RuntimeError("Model loading failed")
        
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(RuntimeError, match="Model loading failed"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_supported_formats_immutability(self):
        """Test that supported formats list cannot be modified externally."""
        adapter = WhisperLocalAdapter(model_name="base")
        formats1 = adapter.get_supported_formats()
        formats2 = adapter.get_supported_formats()
        
        # Should return different list instances
        assert formats1 is not formats2
        
        # Modifying one shouldn't affect the other
        formats1.append("fake_format")
        assert "fake_format" not in formats2
        assert "fake_format" not in adapter.get_supported_formats()

    @patch('whisper.load_model')
    def test_model_loading_with_device_specification(self, mock_load_model):
        """Test that model loading passes device specification correctly."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        adapter = WhisperLocalAdapter(model_name="base", device="cpu")
        adapter._load_model()
        
        # Verify that load_model was called with correct parameters
        mock_load_model.assert_called_once_with("base", device="cpu")
        assert adapter._model_loaded is True
        assert adapter.model is mock_model

    @patch('whisper.load_model')
    def test_lazy_model_loading(self, mock_load_model):
        """Test that model is loaded lazily only when needed."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Model should not be loaded during initialization
        assert adapter._model_loaded is False
        assert mock_load_model.call_count == 0
        
        # Model should be loaded when _ensure_model_loaded is called
        adapter._ensure_model_loaded()
        assert adapter._model_loaded is True
        assert mock_load_model.call_count == 1
        
        # Subsequent calls should not reload the model
        adapter._ensure_model_loaded()
        assert mock_load_model.call_count == 1

    def test_configuration_storage(self):
        """Test that additional configuration is stored correctly."""
        config_kwargs = {
            'timeout': 300,
            'retry_attempts': 3,
            'custom_param': 'test_value'
        }
        
        adapter = WhisperLocalAdapter(model_name="base", **config_kwargs)
        
        assert adapter.config == config_kwargs
        assert adapter.config['timeout'] == 300
        assert adapter.config['retry_attempts'] == 3
        assert adapter.config['custom_param'] == 'test_value'


class TestWhisperLocalAdapterIntegration:
    """Integration tests for WhisperLocalAdapter."""

    @patch('whisper.load_model')
    @patch('whisper.available_models')
    def test_full_adapter_lifecycle(self, mock_available_models, mock_load_model):
        """Test the complete lifecycle of adapter usage."""
        # Mock successful setup
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Hello world',
            'segments': [{'text': 'Hello world', 'start': 0.0, 'end': 1.0}],
            'language': 'en'
        }
        mock_load_model.return_value = mock_model
        mock_available_models.return_value = ['tiny', 'base', 'small', 'medium', 'large']
        
        adapter = WhisperLocalAdapter(model_name="base", device="cpu")
        
        # Validate requirements
        errors = adapter.validate_requirements()
        assert errors == []
        
        # Get engine info
        engine, version = adapter.get_engine_info()
        assert engine == "whisper-local"
        assert version == "base"
        
        # Get supported formats
        formats = adapter.get_supported_formats()
        assert 'mp3' in formats
        
        # Estimate cost (should be free)
        cost = adapter.estimate_cost(60.0)
        assert cost is None
        
        # Create a temporary audio file and test transcription
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            assert result['text'] == 'Hello world'
            assert 'segments' in result
            assert 'language' in result
        finally:
            os.unlink(temp_file_path)

    def test_different_model_sizes(self):
        """Test adapter with different model sizes."""
        model_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        
        for model_size in model_sizes:
            adapter = WhisperLocalAdapter(model_name=model_size)
            
            # Check that model size is set correctly
            assert adapter.model_name == model_size
            
            # Check engine info
            engine, version = adapter.get_engine_info()
            assert engine == "whisper-local"
            assert version == model_size
            
            # Check model size info
            info = adapter.get_model_size_info()
            assert info['model'] == model_size

    def test_different_devices(self):
        """Test adapter with different device specifications."""
        devices = ['cpu', 'cuda', 'auto']
        
        for device in devices:
            adapter = WhisperLocalAdapter(model_name="base", device=device)
            
            # Check that device is set correctly
            assert adapter.device == device
            
            # Check model size info includes device
            info = adapter.get_model_size_info()
            assert info['device'] == device