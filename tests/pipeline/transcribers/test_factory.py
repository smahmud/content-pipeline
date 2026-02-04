"""
File: test_factory.py

Unit tests for the EngineFactory class.

Tests:
- Engine registration and instantiation
- Requirement validation before instantiation
- Configuration passing to adapters
- Error handling for unsupported engines
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipeline.transcribers.factory import EngineFactory
from pipeline.transcription import TranscriberProvider
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.config.schema import TranscriptionConfig, WhisperLocalConfig, WhisperAPIConfig


# Create aliases for backward compatibility with test code
LocalWhisperAdapter = LocalWhisperProvider
OpenAIWhisperAdapter = CloudOpenAIWhisperProvider
TranscriberAdapter = TranscriberProvider


class MockAdapter:
    """Mock adapter for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requirements_errors = []
    
    def transcribe(self, audio_path, language=None):
        return {"text": "mock transcription"}
    
    def get_engine_info(self):
        return ("mock", "1.0")
    
    def validate_requirements(self):
        return self.requirements_errors
    
    def get_supported_formats(self):
        return ["mp3", "wav"]
    
    def estimate_cost(self, duration):
        return None


class TestEngineFactory:
    """Test the EngineFactory class."""
    
    def test_factory_initialization(self):
        """Test that factory initializes with default adapters."""
        factory = EngineFactory()
        
        available_engines = factory.get_available_engines()
        assert 'local-whisper' in available_engines
        assert 'openai-whisper' in available_engines
        assert len(available_engines) >= 2
    
    def test_create_whisper_local_engine(self):
        """Test creating a local Whisper engine."""
        factory = EngineFactory()
        config = TranscriptionConfig(
            engine='local-whisper',
            whisper_local=WhisperLocalConfig(model='base')
        )
        
        with patch.object(LocalWhisperAdapter, 'validate_requirements', return_value=[]):
            adapter = factory.create_engine('local-whisper', config)
            
            assert isinstance(adapter, LocalWhisperAdapter)
            assert adapter.config.model == 'base'
    
    def test_create_engine_with_requirements_failure(self):
        """Test that engine creation fails when requirements are not met."""
        factory = EngineFactory()
        config = TranscriptionConfig(engine='local-whisper')
        
        # Mock requirements validation to return errors
        with patch.object(LocalWhisperAdapter, 'validate_requirements', 
                         return_value=['Whisper not installed', 'Model not found']):
            with pytest.raises(RuntimeError, match="Engine 'local-whisper' requirements not met"):
                factory.create_engine('local-whisper', config)
    
    def test_create_unsupported_engine(self):
        """Test that creating an unsupported engine raises ValueError."""
        factory = EngineFactory()
        config = TranscriptionConfig(engine='unsupported-engine')
        
        with pytest.raises(ValueError, match="Unsupported engine type 'unsupported-engine'"):
            factory.create_engine('unsupported-engine', config)
    
    def test_register_new_adapter(self):
        """Test registering a new adapter type."""
        factory = EngineFactory()
        
        # Register mock adapter
        factory.register_adapter('mock-engine', MockAdapter)
        
        available_engines = factory.get_available_engines()
        assert 'mock-engine' in available_engines
        
        # Test creating the registered adapter
        config = TranscriptionConfig(engine='mock-engine')
        adapter = factory.create_engine('mock-engine', config)
        
        assert isinstance(adapter, MockAdapter)
    
    def test_register_duplicate_adapter_raises_error(self):
        """Test that registering a duplicate adapter raises ValueError."""
        factory = EngineFactory()
        
        with pytest.raises(ValueError, match="Engine type 'local-whisper' is already registered"):
            factory.register_adapter('local-whisper', MockAdapter)
    
    def test_validate_engine_requirements_success(self):
        """Test validating engine requirements when they are met."""
        factory = EngineFactory()
        config = TranscriptionConfig(engine='local-whisper')
        
        with patch.object(LocalWhisperAdapter, 'validate_requirements', return_value=[]):
            errors = factory.validate_engine_requirements('local-whisper', config)
            
            assert errors == []
    
    def test_validate_engine_requirements_failure(self):
        """Test validating engine requirements when they are not met."""
        factory = EngineFactory()
        config = TranscriptionConfig(engine='local-whisper')
        
        expected_errors = ['Whisper not installed', 'Model not available']
        with patch.object(LocalWhisperAdapter, 'validate_requirements', return_value=expected_errors):
            errors = factory.validate_engine_requirements('local-whisper', config)
            
            assert errors == expected_errors
    
    def test_validate_unsupported_engine_requirements(self):
        """Test validating requirements for unsupported engine."""
        factory = EngineFactory()
        config = TranscriptionConfig(engine='unsupported-engine')
        
        errors = factory.validate_engine_requirements('unsupported-engine', config)
        
        assert len(errors) == 1
        assert 'Unsupported engine type' in errors[0]
        assert 'unsupported-engine' in errors[0]
    
    def test_is_engine_available(self):
        """Test checking if engines are available."""
        factory = EngineFactory()
        
        assert factory.is_engine_available('local-whisper') is True
        assert factory.is_engine_available('nonexistent-engine') is False
        
        # Register new engine and test
        factory.register_adapter('test-engine', MockAdapter)
        assert factory.is_engine_available('test-engine') is True
    
    def test_get_engine_info(self):
        """Test getting engine information."""
        factory = EngineFactory()
        
        info = factory.get_engine_info('local-whisper')
        
        assert info['engine_type'] == 'local-whisper'
        assert info['adapter_class'] == 'LocalWhisperProvider'
        assert 'pipeline.transcription.providers.local_whisper' in info['module']
        assert info['is_available'] is True
    
    def test_get_engine_info_unsupported(self):
        """Test getting info for unsupported engine."""
        factory = EngineFactory()
        
        with pytest.raises(ValueError, match="Unsupported engine type 'nonexistent'"):
            factory.get_engine_info('nonexistent')
    
    def test_create_adapter_instance_with_configuration(self):
        """Test that adapter instances receive proper configuration."""
        factory = EngineFactory()
        
        # Test local-whisper configuration
        config = TranscriptionConfig(
            engine='local-whisper',
            whisper_local=WhisperLocalConfig(model='large')
        )
        
        with patch.object(LocalWhisperAdapter, 'validate_requirements', return_value=[]):
            adapter = factory.create_engine('local-whisper', config)
            assert adapter.config.model == 'large'

    def test_create_whisper_api_engine(self):
        """Test creating a Whisper API engine with configuration."""
        factory = EngineFactory()
        
        config = TranscriptionConfig(
            engine='openai-whisper',
            whisper_api=WhisperAPIConfig(
                api_key='sk-test123',
                model='gpt-4o-transcribe',
                temperature=0.1,
                response_format='verbose_json'
            )
        )
        
        with patch.object(OpenAIWhisperAdapter, 'validate_requirements', return_value=[]):
            adapter = factory.create_engine('openai-whisper', config)
            
            assert isinstance(adapter, OpenAIWhisperAdapter)
            assert adapter.config.api_key == 'sk-test123'
            assert adapter.config.model == 'gpt-4o-transcribe'
            assert adapter.config.temperature == 0.1
            assert adapter.config.response_format == 'verbose_json'
    
    # Removed test_get_api_key_from_env - method no longer exists in refactored factory
    # API key handling is now done in the provider configuration classes


class TestEngineFactoryErrorHandling:
    """Test error handling in the engine factory."""
    
    def test_adapter_instantiation_failure(self):
        """Test handling of adapter instantiation failures."""
        factory = EngineFactory()
        
        # Mock adapter class that raises exception during instantiation
        mock_adapter_class = Mock(side_effect=Exception("Instantiation failed"))
        factory.register_adapter('failing-engine', mock_adapter_class)
        
        config = TranscriptionConfig(engine='failing-engine')
        
        errors = factory.validate_engine_requirements('failing-engine', config)
        assert len(errors) == 1
        assert 'Failed to validate engine' in errors[0]
        assert 'Instantiation failed' in errors[0]
    
    def test_requirements_validation_exception(self):
        """Test handling of exceptions during requirements validation."""
        factory = EngineFactory()
        
        # Create mock adapter that raises exception during validation
        mock_adapter = Mock()
        mock_adapter.validate_requirements.side_effect = Exception("Validation error")
        
        mock_adapter_class = Mock(return_value=mock_adapter)
        factory.register_adapter('error-engine', mock_adapter_class)
        
        config = TranscriptionConfig(engine='error-engine')
        
        errors = factory.validate_engine_requirements('error-engine', config)
        assert len(errors) == 1
        assert 'Failed to validate engine' in errors[0]
        assert 'Validation error' in errors[0]
    
    def test_empty_available_engines_list(self):
        """Test behavior when no engines are registered."""
        factory = EngineFactory()
        factory._adapters.clear()  # Remove all registered adapters
        
        available_engines = factory.get_available_engines()
        assert available_engines == []
        
        config = TranscriptionConfig(engine='any-engine')
        with pytest.raises(ValueError, match="Available engines: \\[\\]"):
            factory.create_engine('any-engine', config)


class TestEngineFactoryIntegration:
    """Integration tests for the engine factory."""
    
    def test_full_engine_lifecycle(self):
        """Test the complete lifecycle of engine registration and usage."""
        factory = EngineFactory()
        
        # Register a custom adapter
        factory.register_adapter('custom-engine', MockAdapter)
        
        # Validate it's available
        assert factory.is_engine_available('custom-engine')
        
        # Get engine info
        info = factory.get_engine_info('custom-engine')
        assert info['engine_type'] == 'custom-engine'
        
        # Validate requirements
        config = TranscriptionConfig(engine='custom-engine')
        errors = factory.validate_engine_requirements('custom-engine', config)
        assert errors == []  # MockAdapter returns no errors by default
        
        # Create the engine
        adapter = factory.create_engine('custom-engine', config)
        assert isinstance(adapter, MockAdapter)
        
        # Test the adapter works
        result = adapter.transcribe('test.mp3')
        assert result['text'] == 'mock transcription'
    
    def test_multiple_engine_registration(self):
        """Test registering and using multiple custom engines."""
        factory = EngineFactory()
        
        # Register multiple engines
        factory.register_adapter('engine-1', MockAdapter)
        factory.register_adapter('engine-2', MockAdapter)
        
        available_engines = factory.get_available_engines()
        assert 'engine-1' in available_engines
        assert 'engine-2' in available_engines
        assert 'local-whisper' in available_engines  # Default should still be there
        
        # Create instances of each
        config1 = TranscriptionConfig(engine='engine-1')
        config2 = TranscriptionConfig(engine='engine-2')
        
        adapter1 = factory.create_engine('engine-1', config1)
        adapter2 = factory.create_engine('engine-2', config2)
        
        assert isinstance(adapter1, MockAdapter)
        assert isinstance(adapter2, MockAdapter)
        assert adapter1 is not adapter2  # Should be different instances