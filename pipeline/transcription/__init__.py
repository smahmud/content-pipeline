"""
Transcription Infrastructure Layer

This module provides shared transcription infrastructure for the Content Pipeline.
It extracts transcription provider implementations from domain modules to establish
consistent architecture across all infrastructure services.

Architecture:
    Infrastructure Layer (this module)
        ↓
    Domain Layer (transcribers, extractors)
        ↓
    Application Layer (CLI)

Key Components:
    - providers/: Transcription provider implementations (LocalWhisperProvider, etc.)
    - factory.py: TranscriptionProviderFactory for provider instantiation
    - config.py: Configuration management with environment variable support
    - errors.py: Transcription-specific error classes

Usage:
    >>> from pipeline.transcription import TranscriptionProviderFactory, TranscriptionConfig
    >>> 
    >>> trans_config = TranscriptionConfig.load_from_yaml('.content-pipeline/config.yaml')
    >>> factory = TranscriptionProviderFactory(trans_config)
    >>> provider = factory.create_provider("local-whisper")
    >>> result = provider.transcribe("audio.mp3")

**Validates: Requirement 5.8**
"""

# Export base protocol
from pipeline.transcription.providers.base import TranscriberProvider

# Export factory
from pipeline.transcription.factory import TranscriptionProviderFactory

# Export configuration classes
from pipeline.transcription.config import (
    TranscriptionConfig,
    WhisperLocalConfig,
    WhisperAPIConfig,
    AWSTranscribeConfig
)

# Export error classes
from pipeline.transcription.errors import (
    TranscriptionError,
    ConfigurationError,
    ProviderError,
    ProviderNotAvailableError,
    AudioFileError,
    TranscriptionTimeoutError
)

__all__ = [
    # Base protocol
    "TranscriberProvider",
    
    # Factory
    "TranscriptionProviderFactory",
    
    # Configuration
    "TranscriptionConfig",
    "WhisperLocalConfig",
    "WhisperAPIConfig",
    "AWSTranscribeConfig",
    
    # Errors
    "TranscriptionError",
    "ConfigurationError",
    "ProviderError",
    "ProviderNotAvailableError",
    "AudioFileError",
    "TranscriptionTimeoutError",
]
