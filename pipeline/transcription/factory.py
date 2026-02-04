"""
Transcription Provider Factory

Factory pattern for creating transcription providers with validation support.
Handles provider instantiation, caching, and requirement validation
based on configuration.

**Validates: Requirements 2.4, 9.2, 9.4, 9.5, 9.6**
"""

from typing import Dict, Optional, List

from pipeline.transcription.config import TranscriptionConfig
from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider
from pipeline.transcription.errors import ConfigurationError


class TranscriptionProviderFactory:
    """Factory for creating transcription providers with validation support.
    
    This factory handles:
    - Provider instantiation based on provider name
    - Provider caching to prevent redundant instantiation
    - Provider validation before use
    - Clear error messages for configuration issues
    
    Example:
        >>> from pipeline.transcription.config import TranscriptionConfig
        >>> trans_config = TranscriptionConfig.load_from_yaml('.content-pipeline/config.yaml')
        >>> factory = TranscriptionProviderFactory(trans_config)
        >>> provider = factory.create_provider("local-whisper")
        >>> # Validate before use
        >>> errors = provider.validate_requirements()
        >>> if not errors:
        ...     result = provider.transcribe("audio.mp3")
    """
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize transcription provider factory.
        
        Args:
            config: Transcription configuration with all provider configs
        """
        self.config = config
        
        # Cache for instantiated providers
        self._provider_cache: Dict[str, TranscriberProvider] = {}
    
    def create_provider(self, provider: str) -> TranscriberProvider:
        """Create provider for specified provider name.
        
        Supports provider IDs: local-whisper, cloud-openai-whisper, cloud-aws-transcribe
        
        Args:
            provider: Provider name
            
        Returns:
            Instantiated transcription provider
            
        Raises:
            ConfigurationError: If provider is unknown or unavailable
        """
        # Check cache first
        if provider in self._provider_cache:
            return self._provider_cache[provider]
        
        # Instantiate new provider
        provider_instance = self._instantiate_provider(provider)
        
        # Cache for future use
        self._provider_cache[provider] = provider_instance
        
        return provider_instance
    
    def _instantiate_provider(self, provider: str) -> TranscriberProvider:
        """Instantiate specific provider type.
        
        Args:
            provider: Provider name (local-whisper, cloud-openai-whisper, cloud-aws-transcribe)
            
        Returns:
            Instantiated provider
            
        Raises:
            ConfigurationError: If provider is unknown or not configured
        """
        if provider == "local-whisper":
            return LocalWhisperProvider(self.config.whisper_local)
        
        elif provider == "cloud-openai-whisper":
            if not self.config.whisper_api.api_key:
                raise ConfigurationError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY environment variable or provide in config."
                )
            return CloudOpenAIWhisperProvider(self.config.whisper_api)
        
        elif provider == "cloud-aws-transcribe":
            return CloudAWSTranscribeProvider(self.config.aws_transcribe)
        
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider}. "
                f"Valid options: local-whisper, cloud-openai-whisper, cloud-aws-transcribe"
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently available providers.
        
        Returns:
            List of provider names that pass validation
        """
        available = []
        
        for provider in ["local-whisper", "cloud-openai-whisper", "cloud-aws-transcribe"]:
            try:
                provider_instance = self._instantiate_provider(provider)
                errors = provider_instance.validate_requirements()
                if not errors:
                    available.append(provider)
            except (ConfigurationError, NotImplementedError):
                continue
        
        return available
    
    def validate_provider_requirements(self, provider: str) -> List[str]:
        """Validate that provider requirements are met without caching.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Raises:
            ConfigurationError: If provider is unknown
        """
        try:
            provider_instance = self._instantiate_provider(provider)
            return provider_instance.validate_requirements()
        except ConfigurationError as e:
            return [str(e)]
        except Exception as e:
            return [f"Failed to validate provider '{provider}': {e}"]
    
    def clear_cache(self):
        """Clear the provider cache.
        
        This forces re-instantiation of providers on next request.
        Useful for testing or when configuration changes.
        """
        self._provider_cache.clear()
