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
    >>> from pipeline.transcription.factory import TranscriptionProviderFactory
    >>> from pipeline.transcription.config import TranscriptionConfig
    >>> from pipeline.config.manager import ConfigManager
    >>> 
    >>> config_manager = ConfigManager()
    >>> trans_config = TranscriptionConfig.load_from_config(config_manager)
    >>> factory = TranscriptionProviderFactory(trans_config)
    >>> provider = factory.create_provider("local-whisper")
"""

# Export main components for convenient imports
from pipeline.transcription.providers.base import TranscriberProvider

__all__ = [
    "TranscriberProvider",
]
