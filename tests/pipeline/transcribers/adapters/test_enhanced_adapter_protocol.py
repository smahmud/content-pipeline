"""
File: test_enhanced_adapter_protocol.py

Unit tests for the enhanced TranscriberAdapter protocol.

Tests the new methods added in v0.6.5:
- validate_requirements()
- get_supported_formats()
- estimate_cost()
"""
import pytest
from unittest.mock import patch, MagicMock
from pipeline.transcribers.adapters.whisper_local import WhisperLocalAdapter
from pipeline.transcribers.adapters.base import TranscriberAdapter


class TestEnhancedAdapterProtocol:
    """Test the enhanced adapter protocol methods."""

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

    def test_get_supported_formats_returns_list(self):
        """Test that get_supported_formats returns a list of strings."""
        adapter = WhisperLocalAdapter(model_name="base")
        formats = adapter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        
        # Check for common audio formats
        expected_formats = ['mp3', 'wav', 'flac']
        for fmt in expected_formats:
            assert fmt in formats

    def test_estimate_cost_returns_none_for_local(self):
        """Test that local Whisper returns None for cost estimation."""
        adapter = WhisperLocalAdapter(model_name="base")
        cost = adapter.estimate_cost(60.0)  # 1 minute of audio
        
        assert cost is None  # Local Whisper is free

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
        
        # Create adapter with invalid model (won't fail in constructor now)
        adapter = WhisperLocalAdapter(model_name="invalid_model")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("Local Whisper model 'invalid_model' not available" in error for error in errors)

    def test_validate_requirements_missing_whisper_package(self):
        """Test validate_requirements when whisper package is not installed."""
        # Create adapter first (won't fail without whisper package now)
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Mock import error for whisper package in validate_requirements
        with patch('builtins.__import__', side_effect=ImportError("No module named 'whisper'")):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("OpenAI Whisper package not installed" in error for error in errors)

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
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported audio format"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_get_engine_info_returns_tuple(self):
        """Test that get_engine_info returns a tuple of strings."""
        adapter = WhisperLocalAdapter(model_name="large")
        engine_info = adapter.get_engine_info()
        
        assert isinstance(engine_info, tuple)
        assert len(engine_info) == 2
        assert isinstance(engine_info[0], str)
        assert isinstance(engine_info[1], str)
        assert engine_info[0] == "whisper-local"
        assert engine_info[1] == "large"

    def test_adapter_protocol_compliance(self):
        """Test that WhisperLocalAdapter is recognized as a TranscriberAdapter."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        # This should not raise any type errors
        adapter_protocol: TranscriberAdapter = adapter
        
        # Test that we can call protocol methods
        assert callable(adapter_protocol.transcribe)
        assert callable(adapter_protocol.get_engine_info)
        assert callable(adapter_protocol.validate_requirements)
        assert callable(adapter_protocol.get_supported_formats)
        assert callable(adapter_protocol.estimate_cost)


class TestAdapterErrorHandling:
    """Test error handling in the enhanced adapter."""

    def test_transcribe_with_unloaded_model(self):
        """Test transcribe behavior when model loading fails."""
        adapter = WhisperLocalAdapter(model_name="base")
        
        # Mock the model loading to fail
        with patch.object(adapter, '_load_model', side_effect=RuntimeError("Model loading failed")):
            # Create a temporary valid audio file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                with pytest.raises(RuntimeError, match="Model loading failed"):
                    adapter.transcribe(temp_file_path)
            finally:
                os.unlink(temp_file_path)

    @patch('whisper.load_model')
    def test_model_loading_error_handling(self, mock_load_model):
        """Test error handling during model loading."""
        mock_load_model.side_effect = RuntimeError("CUDA out of memory")
        
        adapter = WhisperLocalAdapter(model_name="large")
        
        # Error should occur during validate_requirements, not construction
        errors = adapter.validate_requirements()
        assert len(errors) > 0
        assert any("Failed to load local Whisper model" in error for error in errors)

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