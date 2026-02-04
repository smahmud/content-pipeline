"""
Unit tests for TranscriptionProviderFactory

Tests provider instantiation, caching, validation, and error handling.

**Validates: Requirement 10.2**
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from pipeline.transcription.factory import TranscriptionProviderFactory
from pipeline.transcription.config import (
    TranscriptionConfig,
    WhisperLocalConfig,
    WhisperAPIConfig,
    AWSTranscribeConfig
)
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider
from pipeline.transcription.errors import ConfigurationError


@pytest.fixture
def transcription_config():
    """Create a test transcription configuration."""
    return TranscriptionConfig(
        whisper_local=WhisperLocalConfig(
            model="base",
            device="cpu"
        ),
        whisper_api=WhisperAPIConfig(
            api_key="sk-test-key",
            model="whisper-1"
        ),
        aws_transcribe=AWSTranscribeConfig(
            access_key_id="AKIA_TEST",
            secret_access_key="test_secret",
            region="us-east-1"
        )
    )


@pytest.fixture
def factory(transcription_config):
    """Create a test factory instance."""
    return TranscriptionProviderFactory(transcription_config)


class TestTranscriptionProviderFactoryInitialization:
    """Test factory initialization."""
    
    def test_factory_initialization(self, transcription_config):
        """Test that factory initializes with configuration."""
        factory = TranscriptionProviderFactory(transcription_config)
        
        assert factory.config == transcription_config
        assert factory._provider_cache == {}
    
    def test_factory_stores_config(self, factory, transcription_config):
        """Test that factory stores the provided configuration."""
        assert factory.config is transcription_config


class TestProviderInstantiation:
    """Test provider instantiation."""
    
    def test_create_local_whisper_provider(self, factory):
        """Test creating local Whisper provider."""
        provider = factory.create_provider("local-whisper")
        
        assert isinstance(provider, LocalWhisperProvider)
        assert provider.config.model == "base"
        assert provider.config.device == "cpu"
    
    def test_create_cloud_openai_whisper_provider(self, factory):
        """Test creating cloud OpenAI Whisper provider."""
        provider = factory.create_provider("cloud-openai-whisper")
        
        assert isinstance(provider, CloudOpenAIWhisperProvider)
        assert provider.config.api_key == "sk-test-key"
        assert provider.config.model == "whisper-1"
    
    def test_create_cloud_aws_transcribe_provider(self, factory):
        """Test creating cloud AWS Transcribe provider."""
        provider = factory.create_provider("cloud-aws-transcribe")
        
        assert isinstance(provider, CloudAWSTranscribeProvider)
        assert provider.config.access_key_id == "AKIA_TEST"
        assert provider.config.region == "us-east-1"
    
    def test_unknown_provider_raises_error(self, factory):
        """Test that unknown provider name raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("unknown-provider")
        
        assert "Unknown provider: unknown-provider" in str(exc_info.value)
        assert "local-whisper" in str(exc_info.value)
        assert "cloud-openai-whisper" in str(exc_info.value)
        assert "cloud-aws-transcribe" in str(exc_info.value)


class TestProviderCaching:
    """Test provider caching behavior."""
    
    def test_provider_is_cached(self, factory):
        """Test that provider is cached after first creation."""
        provider1 = factory.create_provider("local-whisper")
        provider2 = factory.create_provider("local-whisper")
        
        # Should return the same instance
        assert provider1 is provider2
    
    def test_different_providers_cached_separately(self, factory):
        """Test that different providers are cached separately."""
        local_provider = factory.create_provider("local-whisper")
        openai_provider = factory.create_provider("cloud-openai-whisper")
        
        # Should be different instances
        assert local_provider is not openai_provider
        assert isinstance(local_provider, LocalWhisperProvider)
        assert isinstance(openai_provider, CloudOpenAIWhisperProvider)
    
    def test_clear_cache_removes_cached_providers(self, factory):
        """Test that clear_cache removes all cached providers."""
        provider1 = factory.create_provider("local-whisper")
        
        factory.clear_cache()
        
        provider2 = factory.create_provider("local-whisper")
        
        # Should be different instances after cache clear
        assert provider1 is not provider2
    
    def test_cache_persists_across_calls(self, factory):
        """Test that cache persists across multiple calls."""
        provider1 = factory.create_provider("local-whisper")
        provider2 = factory.create_provider("cloud-openai-whisper")
        provider3 = factory.create_provider("local-whisper")
        provider4 = factory.create_provider("cloud-openai-whisper")
        
        assert provider1 is provider3
        assert provider2 is provider4


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_missing_openai_api_key_raises_error(self):
        """Test that missing OpenAI API key raises ConfigurationError."""
        config = TranscriptionConfig(
            whisper_api=WhisperAPIConfig(api_key=None)
        )
        factory = TranscriptionProviderFactory(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("cloud-openai-whisper")
        
        assert "OpenAI API key not configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_empty_openai_api_key_raises_error(self):
        """Test that empty OpenAI API key raises ConfigurationError."""
        config = TranscriptionConfig(
            whisper_api=WhisperAPIConfig(api_key="")
        )
        factory = TranscriptionProviderFactory(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("cloud-openai-whisper")
        
        assert "OpenAI API key not configured" in str(exc_info.value)
    
    def test_aws_transcribe_allows_missing_credentials(self):
        """Test that AWS Transcribe allows missing credentials (uses AWS credential chain)."""
        config = TranscriptionConfig(
            aws_transcribe=AWSTranscribeConfig(
                access_key_id=None,
                secret_access_key=None
            )
        )
        factory = TranscriptionProviderFactory(config)
        
        # Should not raise error - AWS will use credential chain
        provider = factory.create_provider("cloud-aws-transcribe")
        assert isinstance(provider, CloudAWSTranscribeProvider)


class TestProviderValidation:
    """Test provider requirement validation."""
    
    @patch('pipeline.transcription.providers.local_whisper.whisper')
    def test_validate_provider_requirements_success(self, mock_whisper, factory):
        """Test validating provider requirements when all requirements are met."""
        # Mock whisper module
        mock_whisper.available_models.return_value = ['tiny', 'base', 'small', 'medium', 'large']
        mock_whisper.load_model.return_value = Mock()
        
        errors = factory.validate_provider_requirements("local-whisper")
        
        assert errors == []
    
    @patch('pipeline.transcription.providers.local_whisper.whisper', None)
    def test_validate_provider_requirements_failure(self, factory):
        """Test validating provider requirements when requirements are not met."""
        with patch.dict('sys.modules', {'whisper': None}):
            errors = factory.validate_provider_requirements("local-whisper")
            
            assert len(errors) > 0
            assert any("not installed" in error.lower() for error in errors)
    
    def test_validate_unknown_provider(self, factory):
        """Test validating unknown provider returns error."""
        errors = factory.validate_provider_requirements("unknown-provider")
        
        assert len(errors) > 0
        assert "Unknown provider" in errors[0]
    
    @patch('pipeline.transcription.providers.cloud_openai_whisper.CloudOpenAIWhisperProvider.validate_requirements')
    def test_validate_provider_calls_provider_validate(self, mock_validate, factory):
        """Test that validate_provider_requirements calls provider's validate_requirements."""
        mock_validate.return_value = ["test error"]
        
        errors = factory.validate_provider_requirements("cloud-openai-whisper")
        
        mock_validate.assert_called_once()
        assert errors == ["test error"]


class TestGetAvailableProviders:
    """Test getting available providers."""
    
    @patch('pipeline.transcription.providers.local_whisper.whisper')
    @patch('pipeline.transcription.providers.cloud_openai_whisper.CloudOpenAIWhisperProvider.validate_requirements')
    @patch('pipeline.transcription.providers.cloud_aws_transcribe.CloudAWSTranscribeProvider.validate_requirements')
    def test_get_available_providers_all_available(
        self, mock_aws_validate, mock_openai_validate, mock_whisper, factory
    ):
        """Test getting available providers when all are available."""
        # Mock all providers as available
        mock_whisper.available_models.return_value = ['base']
        mock_whisper.load_model.return_value = Mock()
        mock_openai_validate.return_value = []
        mock_aws_validate.return_value = []
        
        available = factory.get_available_providers()
        
        assert "local-whisper" in available
        assert "cloud-openai-whisper" in available
        assert "cloud-aws-transcribe" in available
    
    @patch('pipeline.transcription.providers.local_whisper.whisper', None)
    @patch('pipeline.transcription.providers.cloud_openai_whisper.CloudOpenAIWhisperProvider.validate_requirements')
    @patch('pipeline.transcription.providers.cloud_aws_transcribe.CloudAWSTranscribeProvider.validate_requirements')
    def test_get_available_providers_some_unavailable(
        self, mock_aws_validate, mock_openai_validate, factory
    ):
        """Test getting available providers when some are unavailable."""
        # Mock local whisper as unavailable, others as available
        with patch.dict('sys.modules', {'whisper': None}):
            mock_openai_validate.return_value = []
            mock_aws_validate.return_value = []
            
            available = factory.get_available_providers()
            
            assert "local-whisper" not in available
            assert "cloud-openai-whisper" in available
            assert "cloud-aws-transcribe" in available
    
    def test_get_available_providers_missing_api_key(self):
        """Test that provider with missing API key is not available."""
        config = TranscriptionConfig(
            whisper_api=WhisperAPIConfig(api_key=None)
        )
        factory = TranscriptionProviderFactory(config)
        
        available = factory.get_available_providers()
        
        assert "cloud-openai-whisper" not in available


class TestFactoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_factory_with_minimal_config(self):
        """Test factory with minimal configuration."""
        config = TranscriptionConfig()
        factory = TranscriptionProviderFactory(config)
        
        # Should be able to create local provider with defaults
        provider = factory.create_provider("local-whisper")
        assert isinstance(provider, LocalWhisperProvider)
    
    def test_create_provider_case_sensitive(self, factory):
        """Test that provider names are case-sensitive."""
        with pytest.raises(ConfigurationError):
            factory.create_provider("LOCAL-WHISPER")
        
        with pytest.raises(ConfigurationError):
            factory.create_provider("Local-Whisper")
    
    def test_cache_isolation_between_factories(self, transcription_config):
        """Test that cache is isolated between factory instances."""
        factory1 = TranscriptionProviderFactory(transcription_config)
        factory2 = TranscriptionProviderFactory(transcription_config)
        
        provider1 = factory1.create_provider("local-whisper")
        provider2 = factory2.create_provider("local-whisper")
        
        # Different factory instances should have different caches
        assert provider1 is not provider2
    
    def test_provider_config_matches_factory_config(self, factory):
        """Test that created provider uses config from factory."""
        provider = factory.create_provider("local-whisper")
        
        assert provider.config is factory.config.whisper_local
    
    def test_multiple_provider_types_in_cache(self, factory):
        """Test that cache can hold multiple provider types."""
        local = factory.create_provider("local-whisper")
        openai = factory.create_provider("cloud-openai-whisper")
        aws = factory.create_provider("cloud-aws-transcribe")
        
        assert len(factory._provider_cache) == 3
        assert factory._provider_cache["local-whisper"] is local
        assert factory._provider_cache["cloud-openai-whisper"] is openai
        assert factory._provider_cache["cloud-aws-transcribe"] is aws


class TestFactoryIntegration:
    """Integration tests for factory with real provider instances."""
    
    def test_factory_creates_functional_provider(self, factory):
        """Test that factory creates a functional provider instance."""
        provider = factory.create_provider("local-whisper")
        
        # Provider should have all required methods
        assert hasattr(provider, 'transcribe')
        assert hasattr(provider, 'validate_requirements')
        assert hasattr(provider, 'get_engine_info')
        assert hasattr(provider, 'get_supported_formats')
        assert hasattr(provider, 'estimate_cost')
    
    def test_factory_provider_has_correct_config(self, factory):
        """Test that provider created by factory has correct configuration."""
        provider = factory.create_provider("cloud-openai-whisper")
        
        assert provider.config.api_key == "sk-test-key"
        assert provider.config.model == "whisper-1"
    
    @patch('pipeline.transcription.providers.local_whisper.whisper')
    def test_factory_provider_can_validate(self, mock_whisper, factory):
        """Test that provider created by factory can validate requirements."""
        mock_whisper.available_models.return_value = ['base']
        mock_whisper.load_model.return_value = Mock()
        
        provider = factory.create_provider("local-whisper")
        errors = provider.validate_requirements()
        
        # Should be able to validate
        assert isinstance(errors, list)
