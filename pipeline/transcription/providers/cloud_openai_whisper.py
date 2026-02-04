"""
Cloud OpenAI Whisper Transcription Provider

Implements the CloudOpenAIWhisperProvider using OpenAI's Whisper API.
Conforms to the TranscriberProvider protocol.

This provider processes audio using OpenAI's cloud-based Whisper API for high-quality transcription.
Migrated from pipeline.transcribers.adapters.openai_whisper as part of the infrastructure
refactoring to establish clean separation between infrastructure and domain layers.

**Validates: Requirements 2.2, 2.3**
"""
from typing import Optional, List, Tuple
import os
from pathlib import Path

from pipeline.utils.retry import retry
from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.transcription.config import WhisperAPIConfig
from pipeline.transcription.errors import (
    AudioFileError,
    ProviderError,
    ProviderNotAvailableError,
    ConfigurationError
)


class CloudOpenAIWhisperProvider(TranscriberProvider):
    """
    Transcribes audio using OpenAI's Whisper API.
    
    This provider provides high-quality transcription using OpenAI's cloud-based service.
    Supports requirement validation, format checking, and cost estimation.
    
    Configuration is provided via WhisperAPIConfig, which supports:
    - API key configuration
    - Model selection (whisper-1, etc.)
    - Temperature and response format settings
    - Timeout and retry configuration
    
    Example:
        >>> from pipeline.transcription.config import WhisperAPIConfig
        >>> config = WhisperAPIConfig(api_key="sk-...", model="whisper-1")
        >>> provider = CloudOpenAIWhisperProvider(config)
        >>> result = provider.transcribe("audio.mp3", language="en")
        >>> print(result['text'])
    """
    
    # Supported audio formats by OpenAI Whisper API
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
    
    # API pricing (as of 2024) - $0.006 per minute
    COST_PER_MINUTE = 0.006
    
    # Maximum file size (25 MB)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes
    
    def __init__(self, config: WhisperAPIConfig):
        """
        Initialize the OpenAI Whisper API provider with configuration.
        
        Args:
            config: WhisperAPIConfig instance with provider configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, WhisperAPIConfig):
            raise ConfigurationError(
                f"Expected WhisperAPIConfig, got {type(config).__name__}"
            )
        
        self.config = config
        self.client = None

    def _ensure_client_initialized(self):
        """Ensure the OpenAI client is initialized."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ProviderNotAvailableError(
                    "OpenAI package not installed. Install with: pip install openai"
                )

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file using OpenAI Whisper API.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (ISO 639-1 code, e.g., 'en', 'es')
            
        Returns:
            Raw transcript dictionary from OpenAI Whisper API containing:
                - text: The transcribed text
                - segments: List of segment dictionaries (if available)
                - language: Detected or specified language
                - duration: Audio duration (if verbose_json format)
                - words: Word-level timestamps (if verbose_json format)
            
        Raises:
            AudioFileError: If audio file doesn't exist, format is not supported, or file is too large
            ProviderError: If transcription fails
            ProviderNotAvailableError: If client is not initialized
        """
        if not os.path.exists(audio_path):
            raise AudioFileError(f"Audio file not found: {audio_path}")
            
        # Check if file format is supported
        file_ext = Path(audio_path).suffix.lower().lstrip('.')
        if file_ext not in self.SUPPORTED_FORMATS:
            raise AudioFileError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size > self.MAX_FILE_SIZE:
            raise AudioFileError(
                f"File too large: {file_size / (1024*1024):.1f}MB. "
                f"Maximum size: {self.MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Ensure client is initialized
        self._ensure_client_initialized()
        
        if self.client is None:
            raise ProviderNotAvailableError(
                "OpenAI client not initialized. Check API key and validate_requirements()."
            )
            
        try:
            # Prepare transcription parameters
            transcription_params = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "response_format": self.config.response_format
            }
            
            # Add language if specified
            if language:
                transcription_params["language"] = language
            
            # Open and transcribe the audio file
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    **transcription_params
                )
            
            # Convert response to dictionary format compatible with local Whisper
            if self.config.response_format == "json":
                return {
                    "text": response.text,
                    "segments": getattr(response, 'segments', []),
                    "language": getattr(response, 'language', language or 'unknown')
                }
            elif self.config.response_format == "verbose_json":
                return {
                    "text": response.text,
                    "segments": response.segments,
                    "language": response.language,
                    "duration": getattr(response, 'duration', None),
                    "words": getattr(response, 'words', [])
                }
            else:
                # For text, srt, vtt formats, wrap in compatible structure
                return {
                    "text": str(response),
                    "segments": [],
                    "language": language or 'unknown'
                }
                
        except Exception as e:
            raise ProviderError(f"OpenAI Whisper API transcription failed: {e}")

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the provider name and model variant.
        
        Returns:
            Tuple of (provider_name, model_variant)
            
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> name, version = provider.get_engine_info()
            >>> print(f"{name} using {version}")
            "cloud-openai-whisper using whisper-1"
        """
        return ("cloud-openai-whisper", self.config.model)

    def validate_requirements(self) -> List[str]:
        """
        Validate that OpenAI API is accessible and properly configured.
        
        This method checks:
        - API key is configured
        - OpenAI package is installed
        - API key format is valid
        - Client can be initialized
        
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> errors = provider.validate_requirements()
            >>> if errors:
            ...     print("Provider not ready:", errors)
            ... else:
            ...     print("Provider ready to use")
        """
        errors = []
        
        # Check if API key is available first (before trying to import)
        if self.config.api_key is None or not self.config.api_key:
            errors.append(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or configure api_key in config.yaml"
            )
            return errors  # Can't continue without API key
        
        # Check if openai package is available
        try:
            import openai
        except ImportError:
            errors.append("OpenAI package not installed. Install with: pip install openai")
            return errors  # Can't continue without the package
        
        # Validate API key format (should start with 'sk-')
        if not self.config.api_key.startswith('sk-'):
            errors.append("Invalid OpenAI API key format. API key should start with 'sk-'.")
        
        # Test API connectivity (optional - can be expensive)
        try:
            self._ensure_client_initialized()
            # We could test with a minimal API call here, but it would cost money
            # For now, just ensure client can be created
        except Exception as e:
            errors.append(f"Failed to initialize OpenAI client: {e}")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats for OpenAI Whisper API.
        
        Returns:
            List of supported file extensions
            
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> formats = provider.get_supported_formats()
            >>> print("Supported:", ", ".join(formats))
            "Supported: mp3, mp4, mpeg, mpga, m4a, wav, webm"
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for OpenAI Whisper API.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD
            
        Note:
            Cost is calculated using the configured cost_per_minute_usd rate.
            Default rate is $0.006/minute but can be overridden in configuration
            for enterprise customers with custom pricing.
            
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> cost = provider.estimate_cost(300.0)  # 5 minutes
            >>> print(f"Estimated cost: ${cost:.4f}")
            "Estimated cost: $0.0300"
        """
        if audio_duration <= 0:
            return 0.0
        
        # Convert seconds to minutes and calculate cost using config value
        duration_minutes = audio_duration / 60.0
        return round(duration_minutes * self.config.cost_per_minute_usd, 4)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model and configuration.
        
        This is a provider-specific method that provides additional information
        about the model being used and its configuration.
        
        Returns:
            Dictionary with model information:
                - model: Model name
                - temperature: Sampling temperature
                - response_format: Response format
                - max_file_size_mb: Maximum file size in MB
                - cost_per_minute_usd: Cost per minute in USD
                - api_key_configured: Whether API key is configured
                
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> info = provider.get_model_info()
            >>> print(f"Model: {info['model']}, Cost: ${info['cost_per_minute_usd']}/min")
            "Model: whisper-1, Cost: $0.006/min"
        """
        return {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'response_format': self.config.response_format,
            'max_file_size_mb': self.MAX_FILE_SIZE / (1024 * 1024),
            'cost_per_minute_usd': self.config.cost_per_minute_usd,
            'api_key_configured': bool(self.config.api_key)
        }
    
    def get_file_size_limit(self) -> int:
        """
        Get the maximum file size limit in bytes.
        
        This is a provider-specific method that returns the maximum file size
        that can be processed by the OpenAI Whisper API.
        
        Returns:
            Maximum file size in bytes
            
        Example:
            >>> provider = CloudOpenAIWhisperProvider(config)
            >>> limit = provider.get_file_size_limit()
            >>> print(f"Max file size: {limit / (1024*1024)}MB")
            "Max file size: 25.0MB"
        """
        return self.MAX_FILE_SIZE
