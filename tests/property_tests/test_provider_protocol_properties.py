"""
Property-based tests for transcription provider protocol conformance.

**Property 4: Transcription Provider Protocol Conformance**
*For any* transcription provider implementation, it should conform to the base TranscriberProvider 
protocol and return standardized transcript formats.
**Validates: Requirements 2.5, 3.6, 10.3**
"""

import pytest
import tempfile
import os
from typing import List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings

from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.factory import TranscriptionProviderFactory
from pipeline.transcription.config import WhisperLocalConfig, WhisperAPIConfig
from pipeline.config.schema import TranscriptionConfig


# Strategy for generating valid provider instances
def create_whisper_local_provider():
    """Create a LocalWhisperProvider for testing."""
    config = WhisperLocalConfig(model="base", device="cpu")
    return LocalWhisperProvider(config)

def create_whisper_api_provider():
    """Create a CloudOpenAIWhisperProvider for testing."""
    config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
    return CloudOpenAIWhisperProvider(config)

# Strategy for provider instances
adapter_strategies = st.one_of(
    st.just(create_whisper_local_provider()),
    st.just(create_whisper_api_provider())
)

# Strategy for valid audio file extensions
valid_audio_extensions = st.sampled_from(['mp3', 'wav', 'm4a', 'webm', 'mp4'])

# Strategy for language codes (ISO 639-1)
language_codes = st.sampled_from(['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'])

# Strategy for audio durations (in seconds)
audio_durations = st.floats(min_value=0.1, max_value=3600.0)  # 0.1 seconds to 1 hour


class TestProviderProtocolConformance:
    """Test that all providers conform to the TranscriberProvider protocol."""

    def test_whisper_local_provider_protocol_conformance(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        LocalWhisperProvider should implement all required protocol methods.
        """
        config = WhisperLocalConfig(model="base", device="cpu")
        adapter = LocalWhisperProvider(config)
        
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

    def test_whisper_api_provider_protocol_conformance(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        CloudOpenAIWhisperProvider should implement all required protocol methods.
        """
        config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        adapter = CloudOpenAIWhisperProvider(config)
        
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

    def test_get_engine_info_returns_correct_format(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, get_engine_info should return a tuple of (engine_name, model_variant).
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_info = local_adapter.get_engine_info()
        
        assert isinstance(local_info, tuple)
        assert len(local_info) == 2
        assert isinstance(local_info[0], str)  # engine_name
        assert isinstance(local_info[1], str)  # model_variant
        assert local_info[0] == "local-whisper"
        assert local_info[1] == "base"
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_info = api_adapter.get_engine_info()
        
        assert isinstance(api_info, tuple)
        assert len(api_info) == 2
        assert isinstance(api_info[0], str)  # engine_name
        assert isinstance(api_info[1], str)  # model_variant
        assert api_info[0] == "cloud-openai-whisper"
        assert api_info[1] == "whisper-1"

    def test_validate_requirements_returns_list_of_strings(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, validate_requirements should return a list of error strings.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_errors = local_adapter.validate_requirements()
        
        assert isinstance(local_errors, list)
        assert all(isinstance(error, str) for error in local_errors)
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_errors = api_adapter.validate_requirements()
        
        assert isinstance(api_errors, list)
        assert all(isinstance(error, str) for error in api_errors)

    def test_get_supported_formats_returns_list_of_strings(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, get_supported_formats should return a list of format strings.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_formats = local_adapter.get_supported_formats()
        
        assert isinstance(local_formats, list)
        assert len(local_formats) > 0
        assert all(isinstance(fmt, str) for fmt in local_formats)
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_formats = api_adapter.get_supported_formats()
        
        assert isinstance(api_formats, list)
        assert len(api_formats) > 0
        assert all(isinstance(fmt, str) for fmt in api_formats)

    @given(duration=audio_durations)
    def test_estimate_cost_returns_valid_number_or_none(self, duration):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* valid audio duration, estimate_cost should return a number or None.
        """
        # Test LocalWhisperProvider (should return None for free engines)
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_cost = local_adapter.estimate_cost(duration)
        
        assert local_cost is None or isinstance(local_cost, (int, float))
        if local_cost is not None:
            assert local_cost >= 0
        
        # Test CloudOpenAIWhisperProvider (should return a positive number)
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_cost = api_adapter.estimate_cost(duration)
        
        assert isinstance(api_cost, (int, float))
        assert api_cost >= 0
        
        # Cost should be proportional to duration for non-zero durations
        # Note: Very small durations may round to 0.0 due to pricing precision
        if duration >= 60.0:  # Only check for durations >= 1 minute to avoid rounding issues
            assert api_cost > 0

    def test_estimate_cost_zero_duration(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, zero duration should result in zero cost.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_cost = local_adapter.estimate_cost(0.0)
        assert local_cost is None or local_cost == 0.0
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_cost = api_adapter.estimate_cost(0.0)
        assert api_cost == 0.0

    def test_estimate_cost_negative_duration(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, negative duration should result in zero cost.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_cost = local_adapter.estimate_cost(-10.0)
        assert local_cost is None or local_cost == 0.0
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_cost = api_adapter.estimate_cost(-10.0)
        assert api_cost == 0.0

    @patch('pipeline.transcription.providers.local_whisper.whisper.load_model')
    def test_transcribe_returns_standardized_format_local(self, mock_load_model):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* successful transcription, LocalWhisperProvider should return standardized format.
        """
        # Mock the Whisper model and transcription result
        mock_model = Mock()
        mock_result = {
            'text': 'Hello world',
            'segments': [{'text': 'Hello world', 'start': 0.0, 'end': 1.0}],
            'language': 'en'
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        
        config = WhisperLocalConfig(model="base", device="cpu")
        adapter = LocalWhisperProvider(config)
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            
            # Check standardized format
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'segments' in result
            assert 'language' in result
            
            assert isinstance(result['text'], str)
            assert isinstance(result['segments'], list)
            assert isinstance(result['language'], str)
            
        finally:
            os.unlink(temp_file_path)

    @patch('openai.OpenAI')
    def test_transcribe_returns_standardized_format_api(self, mock_openai):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* successful transcription, CloudOpenAIWhisperProvider should return standardized format.
        """
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = []
        mock_response.language = "en"
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        adapter = CloudOpenAIWhisperProvider(config)
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            result = adapter.transcribe(temp_file_path)
            
            # Check standardized format
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'segments' in result
            assert 'language' in result
            
            assert isinstance(result['text'], str)
            assert isinstance(result['segments'], list)
            assert isinstance(result['language'], str)
            
        finally:
            os.unlink(temp_file_path)

    def test_transcribe_file_not_found_error(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, transcribing a non-existent file should raise AudioFileError.
        """
        from pipeline.transcription.errors import AudioFileError
        
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        with pytest.raises(AudioFileError):
            local_adapter.transcribe("nonexistent_file.mp3")
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        with pytest.raises(AudioFileError):
            api_adapter.transcribe("nonexistent_file.mp3")

    @given(extension=valid_audio_extensions)
    def test_supported_formats_include_common_formats(self, extension):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, supported formats should include common audio formats.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        local_formats = local_adapter.get_supported_formats()
        
        # Should support common formats
        common_formats = ['mp3', 'wav', 'm4a']
        for fmt in common_formats:
            assert fmt in local_formats
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_formats = api_adapter.get_supported_formats()
        
        # Should support OpenAI API formats
        api_common_formats = ['mp3', 'wav', 'm4a', 'webm']
        for fmt in api_common_formats:
            assert fmt in api_formats

    def test_supported_formats_immutability(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, get_supported_formats should return a new list each time.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        formats1 = local_adapter.get_supported_formats()
        formats2 = local_adapter.get_supported_formats()
        
        # Should return different list instances
        assert formats1 is not formats2
        
        # Modifying one shouldn't affect the other
        formats1.append("fake_format")
        assert "fake_format" not in formats2
        assert "fake_format" not in local_adapter.get_supported_formats()
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        api_formats1 = api_adapter.get_supported_formats()
        api_formats2 = api_adapter.get_supported_formats()
        
        # Should return different list instances
        assert api_formats1 is not api_formats2
        
        # Modifying one shouldn't affect the other
        api_formats1.append("fake_format")
        assert "fake_format" not in api_formats2
        assert "fake_format" not in api_adapter.get_supported_formats()

    def test_factory_creates_protocol_compliant_providers(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider created by the factory, it should be protocol compliant.
        """
        # Test local-whisper
        local_config = TranscriptionConfig(
            engine='local-whisper',
            whisper_local=WhisperLocalConfig(model='base', device='cpu')
        )
        factory = TranscriptionProviderFactory(local_config)
        
        with patch('pipeline.transcription.providers.local_whisper.LocalWhisperProvider.validate_requirements', return_value=[]), \
             patch('pipeline.transcription.providers.local_whisper.LocalWhisperProvider._load_model'):
            local_adapter = factory.create_provider('local-whisper')
            
            # Should be protocol compliant
            assert isinstance(local_adapter, TranscriberProvider)
            assert hasattr(local_adapter, 'transcribe')
            assert hasattr(local_adapter, 'get_engine_info')
            assert hasattr(local_adapter, 'validate_requirements')
            assert hasattr(local_adapter, 'get_supported_formats')
            assert hasattr(local_adapter, 'estimate_cost')
        
        # Test cloud-openai-whisper
        api_config = TranscriptionConfig(
            engine='cloud-openai-whisper',
            whisper_api=WhisperAPIConfig(api_key='sk-test123', model='whisper-1')
        )
        factory = TranscriptionProviderFactory(api_config)
        
        with patch('pipeline.transcription.providers.cloud_openai_whisper.CloudOpenAIWhisperProvider.validate_requirements', return_value=[]):
            api_adapter = factory.create_provider('cloud-openai-whisper')
            
            # Should be protocol compliant
            assert isinstance(api_adapter, TranscriberProvider)
            assert hasattr(api_adapter, 'transcribe')
            assert hasattr(api_adapter, 'get_engine_info')
            assert hasattr(api_adapter, 'validate_requirements')
            assert hasattr(api_adapter, 'get_supported_formats')
            assert hasattr(api_adapter, 'estimate_cost')

    @given(language=st.one_of(st.none(), language_codes))
    def test_transcribe_language_parameter_handling(self, language):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* language parameter (including None), providers should handle it gracefully.
        """
        # This test verifies that the language parameter is properly handled
        # We'll mock the actual transcription to avoid dependencies
        
        # Test LocalWhisperProvider
        with patch('pipeline.transcription.providers.local_whisper.whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_result = {
                'text': 'Hello world',
                'segments': [],
                'language': language or 'unknown'
            }
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            local_config = WhisperLocalConfig(model="base", device="cpu")
            local_adapter = LocalWhisperProvider(local_config)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                # Should not raise an exception regardless of language parameter
                result = local_adapter.transcribe(temp_file_path, language=language)
                assert isinstance(result, dict)
                assert 'text' in result
                assert 'language' in result
            finally:
                os.unlink(temp_file_path)
        
        # Test CloudOpenAIWhisperProvider
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello world"
            mock_response.segments = []
            mock_response.language = language or 'unknown'
            
            mock_client.audio.transcriptions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
            api_adapter = CloudOpenAIWhisperProvider(api_config)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                # Should not raise an exception regardless of language parameter
                result = api_adapter.transcribe(temp_file_path, language=language)
                assert isinstance(result, dict)
                assert 'text' in result
                assert 'language' in result
            finally:
                os.unlink(temp_file_path)

    def test_adapter_protocol_type_checking(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, it should be recognized as implementing TranscriberProvider.
        """
        # Test LocalWhisperProvider
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        assert isinstance(local_adapter, TranscriberProvider)
        
        # Test CloudOpenAIWhisperProvider
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        assert isinstance(api_adapter, TranscriberProvider)

    def test_error_handling_consistency(self):
        """
        **Property 4: Transcription Provider Protocol Conformance**
        *For any* provider, error handling should be consistent and informative.
        """
        from pipeline.transcription.errors import AudioFileError
        
        # Test that both providers handle invalid file formats consistently
        local_config = WhisperLocalConfig(model="base", device="cpu")
        local_adapter = LocalWhisperProvider(local_config)
        
        api_config = WhisperAPIConfig(api_key="sk-test123", model="whisper-1")
        api_adapter = CloudOpenAIWhisperProvider(api_config)
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            # Both should raise AudioFileError for unsupported formats
            with pytest.raises(AudioFileError):
                local_adapter.transcribe(temp_file_path)
            
            with pytest.raises(AudioFileError):
                api_adapter.transcribe(temp_file_path)
                
        finally:
            os.unlink(temp_file_path)
