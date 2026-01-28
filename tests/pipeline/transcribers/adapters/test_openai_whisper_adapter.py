"""
File: test_whisper_api_adapter.py

Unit tests for OpenAIWhisperAdapter behavior and error handling.

Covers:
- OpenAI Whisper API transcription functionality
- API key validation and authentication
- Cost estimation and file size limits
- Enhanced error handling and validation
- Protocol compliance and configuration
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipeline.transcribers.adapters.openai_whisper import OpenAIWhisperAdapter
from pipeline.transcribers.adapters.base import TranscriberAdapter


class TestOpenAIWhisperAdapter:
    """Test the OpenAIWhisperAdapter class."""

    def test_adapter_implements_enhanced_protocol(self):
        """Test that OpenAIWhisperAdapter implements all enhanced protocol methods."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
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
        """Test that OpenAIWhisperAdapter is recognized as a TranscriberAdapter."""
        adapter: TranscriberAdapter = OpenAIWhisperAdapter(api_key="sk-test123")
        engine, version = adapter.get_engine_info()
        assert isinstance(engine, str)
        assert isinstance(version, str)
        assert engine == "openai-whisper"

    def test_initialization_with_default_parameters(self):
        """Test adapter initialization with default parameters."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}):
            adapter = OpenAIWhisperAdapter()
            
            assert adapter.api_key == "sk-test123"
            assert adapter.model == "whisper-1"
            assert adapter.temperature == 0.0
            assert adapter.response_format == "json"
            assert adapter.client is None

    def test_initialization_with_custom_parameters(self):
        """Test adapter initialization with custom parameters."""
        adapter = OpenAIWhisperAdapter(
            api_key="sk-custom123",
            model="gpt-4o-transcribe",
            temperature=0.2,
            response_format="verbose_json"
        )
        
        assert adapter.api_key == "sk-custom123"
        assert adapter.model == "gpt-4o-transcribe"
        assert adapter.temperature == 0.2
        assert adapter.response_format == "verbose_json"
        assert adapter.client is None

    def test_get_api_key_from_environment(self):
        """Test API key retrieval from environment variables."""
        # Test OPENAI_API_KEY
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-env123'}):
            adapter = OpenAIWhisperAdapter()
            assert adapter.api_key == "sk-env123"
        
        # Test CONTENT_PIPELINE_OPENAI_API_KEY
        with patch.dict(os.environ, {'CONTENT_PIPELINE_OPENAI_API_KEY': 'sk-pipeline123'}, clear=True):
            adapter = OpenAIWhisperAdapter()
            assert adapter.api_key == "sk-pipeline123"
        
        # Test no environment variable
        with patch.dict(os.environ, {}, clear=True):
            adapter = OpenAIWhisperAdapter()
            assert adapter.api_key is None

    def test_get_supported_formats_returns_list(self):
        """Test that get_supported_formats returns a list of strings."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        formats = adapter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        
        # Check for expected API formats
        expected_formats = ['mp3', 'wav', 'm4a', 'webm']
        for fmt in expected_formats:
            assert fmt in formats

    def test_estimate_cost_calculation(self):
        """Test cost estimation calculation."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Test 1 minute of audio
        cost = adapter.estimate_cost(60.0)
        assert cost == 0.006  # $0.006 per minute
        
        # Test 30 seconds of audio
        cost = adapter.estimate_cost(30.0)
        assert cost == 0.003  # Half a minute
        
        # Test 2.5 minutes of audio
        cost = adapter.estimate_cost(150.0)
        assert cost == 0.015  # 2.5 * $0.006
        
        # Test zero duration
        cost = adapter.estimate_cost(0.0)
        assert cost == 0.0
        
        # Test negative duration
        cost = adapter.estimate_cost(-10.0)
        assert cost == 0.0

    def test_get_engine_info_returns_correct_info(self):
        """Test that get_engine_info returns correct engine information."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123", model="gpt-4o-transcribe")
        engine_info = adapter.get_engine_info()
        
        assert isinstance(engine_info, tuple)
        assert len(engine_info) == 2
        assert engine_info[0] == "openai-whisper"
        assert engine_info[1] == "gpt-4o-transcribe"

    def test_get_model_info(self):
        """Test that get_model_info returns model information."""
        adapter = OpenAIWhisperAdapter(
            api_key="sk-test123",
            model="whisper-1",
            temperature=0.1,
            response_format="verbose_json"
        )
        info = adapter.get_model_info()
        
        assert isinstance(info, dict)
        assert info['model'] == 'whisper-1'
        assert info['temperature'] == 0.1
        assert info['response_format'] == 'verbose_json'
        assert info['max_file_size_mb'] == 25
        assert info['cost_per_minute_usd'] == 0.006
        assert info['api_key_configured'] is True

    def test_get_file_size_limit(self):
        """Test file size limit retrieval."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        limit = adapter.get_file_size_limit()
        
        assert limit == 25 * 1024 * 1024  # 25 MB in bytes

    @patch('openai.OpenAI')
    def test_validate_requirements_success(self, mock_openai):
        """Test validate_requirements when all requirements are met."""
        # Mock successful OpenAI client creation
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) == 0  # No errors when everything is working

    def test_validate_requirements_missing_openai_package(self):
        """Test validate_requirements when openai package is not installed."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Mock import error for openai package
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("OpenAI package not installed" in error for error in errors)

    def test_validate_requirements_missing_api_key(self):
        """Test validate_requirements when API key is missing."""
        adapter = OpenAIWhisperAdapter(api_key=None)
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("OpenAI API key not found" in error for error in errors)

    def test_validate_requirements_invalid_api_key_format(self):
        """Test validate_requirements with invalid API key format."""
        adapter = OpenAIWhisperAdapter(api_key="invalid-key-format")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("Invalid OpenAI API key format" in error for error in errors)

    @patch('openai.OpenAI')
    def test_validate_requirements_client_initialization_failure(self, mock_openai):
        """Test validate_requirements when client initialization fails."""
        mock_openai.side_effect = Exception("API connection failed")
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("Failed to initialize OpenAI client" in error for error in errors)

    def test_transcribe_validates_file_existence(self):
        """Test that transcribe method validates file existence."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            adapter.transcribe("nonexistent_file.mp3")

    def test_transcribe_validates_file_format(self):
        """Test that transcribe method validates file format."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
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

    def test_transcribe_validates_file_size(self):
        """Test that transcribe method validates file size."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Create a temporary file that's too large
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            # Write more than 25MB of data
            large_data = b"x" * (26 * 1024 * 1024)  # 26 MB
            temp_file.write(large_data)
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="File too large"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_with_json_response(self, mock_openai):
        """Test transcribe with JSON response format."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = []
        mock_response.language = "en"
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123", response_format="json")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            
            assert result['text'] == 'Hello world'
            assert 'segments' in result
            assert result['language'] == 'en'
            
            # Verify API was called with correct parameters
            mock_client.audio.transcriptions.create.assert_called_once()
            call_args = mock_client.audio.transcriptions.create.call_args
            assert call_args[1]['model'] == 'whisper-1'
            assert call_args[1]['temperature'] == 0.0
            assert call_args[1]['response_format'] == 'json'
            
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_with_verbose_json_response(self, mock_openai):
        """Test transcribe with verbose JSON response format."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = [{"text": "Hello world", "start": 0.0, "end": 1.0}]
        mock_response.language = "en"
        mock_response.duration = 1.0
        mock_response.words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123", response_format="verbose_json")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            
            assert result['text'] == 'Hello world'
            assert result['segments'] == [{"text": "Hello world", "start": 0.0, "end": 1.0}]
            assert result['language'] == 'en'
            assert result['duration'] == 1.0
            assert result['words'] == [{"word": "Hello", "start": 0.0, "end": 0.5}]
            
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_with_text_response(self, mock_openai):
        """Test transcribe with text response format."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_client.audio.transcriptions.create.return_value = "Hello world"
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123", response_format="text")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            
            assert result['text'] == 'Hello world'
            assert result['segments'] == []
            assert result['language'] == 'unknown'
            
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_with_language_parameter(self, mock_openai):
        """Test transcribe with language parameter."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Bonjour le monde"
        mock_response.language = "fr"
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path, language="fr")
            
            assert result['text'] == 'Bonjour le monde'
            assert result['language'] == 'fr'
            
            # Verify language was passed to API
            call_args = mock_client.audio.transcriptions.create.call_args
            assert call_args[1]['language'] == 'fr'
            
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_api_error_handling(self, mock_openai):
        """Test transcribe error handling when API fails."""
        # Mock OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API rate limit exceeded")
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(RuntimeError, match="OpenAI Whisper API transcription failed"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_supported_formats_immutability(self):
        """Test that supported formats list cannot be modified externally."""
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        formats1 = adapter.get_supported_formats()
        formats2 = adapter.get_supported_formats()
        
        # Should return different list instances
        assert formats1 is not formats2
        
        # Modifying one shouldn't affect the other
        formats1.append("fake_format")
        assert "fake_format" not in formats2
        assert "fake_format" not in adapter.get_supported_formats()

    def test_configuration_storage(self):
        """Test that additional configuration is stored correctly."""
        config_kwargs = {
            'timeout': 300,
            'retry_attempts': 3,
            'custom_param': 'test_value'
        }
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123", **config_kwargs)
        
        assert adapter.config == config_kwargs
        assert adapter.config['timeout'] == 300
        assert adapter.config['retry_attempts'] == 3
        assert adapter.config['custom_param'] == 'test_value'


class TestOpenAIWhisperAdapterIntegration:
    """Integration tests for OpenAIWhisperAdapter."""

    @patch('openai.OpenAI')
    def test_full_adapter_lifecycle(self, mock_openai):
        """Test the complete lifecycle of adapter usage."""
        # Mock successful setup
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = 'Hello world'
        mock_response.segments = [{'text': 'Hello world', 'start': 0.0, 'end': 1.0}]
        mock_response.language = 'en'
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        adapter = OpenAIWhisperAdapter(api_key="sk-test123")
        
        # Validate requirements
        errors = adapter.validate_requirements()
        assert errors == []
        
        # Get engine info
        engine, version = adapter.get_engine_info()
        assert engine == "openai-whisper"
        assert version == "whisper-1"
        
        # Get supported formats
        formats = adapter.get_supported_formats()
        assert 'mp3' in formats
        
        # Estimate cost
        cost = adapter.estimate_cost(60.0)
        assert cost == 0.006
        
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

    def test_different_models(self):
        """Test adapter with different model configurations."""
        models = ['whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe']
        
        for model in models:
            adapter = OpenAIWhisperAdapter(api_key="sk-test123", model=model)
            
            # Check that model is set correctly
            assert adapter.model == model
            
            # Check engine info
            engine, version = adapter.get_engine_info()
            assert engine == "openai-whisper"
            assert version == model
            
            # Check model info
            info = adapter.get_model_info()
            assert info['model'] == model

    def test_different_response_formats(self):
        """Test adapter with different response formats."""
        formats = ['json', 'text', 'srt', 'verbose_json', 'vtt']
        
        for response_format in formats:
            adapter = OpenAIWhisperAdapter(api_key="sk-test123", response_format=response_format)
            
            # Check that format is set correctly
            assert adapter.response_format == response_format
            
            # Check model info includes format
            info = adapter.get_model_info()
            assert info['response_format'] == response_format