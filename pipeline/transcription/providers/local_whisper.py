"""
Local Whisper Transcription Provider

Implements the LocalWhisperProvider using OpenAI's local Whisper model.
Conforms to the TranscriberProvider protocol.

This provider processes audio entirely on the local machine for privacy-focused transcription.
Migrated from pipeline.transcribers.adapters.local_whisper as part of the infrastructure
refactoring to establish clean separation between infrastructure and domain layers.

**Validates: Requirements 2.2, 2.3**
"""
from typing import Optional, List, Tuple
import whisper
import os
from pathlib import Path

from pipeline.utils.retry import retry
from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.transcription.config import WhisperLocalConfig
from pipeline.transcription.errors import (
    AudioFileError,
    ProviderError,
    ProviderNotAvailableError,
    ConfigurationError
)


class LocalWhisperProvider(TranscriberProvider):
    """
    Transcribes audio using a locally loaded Whisper model.
    
    This provider ensures complete privacy by processing audio entirely on the local machine.
    Supports requirement validation, format checking, and improved error handling.
    
    Configuration is provided via WhisperLocalConfig, which supports:
    - Model selection (tiny, base, small, medium, large)
    - Device selection (cpu, cuda, auto)
    - Compute type for faster-whisper
    - Timeout and retry configuration
    
    Example:
        >>> from pipeline.transcription.config import WhisperLocalConfig
        >>> config = WhisperLocalConfig(model="base", device="auto")
        >>> provider = LocalWhisperProvider(config)
        >>> result = provider.transcribe("audio.mp3", language="en")
        >>> print(result['text'])
    """
    
    # Supported audio formats by local Whisper
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'wma', 'aac']
    
    def __init__(self, config: WhisperLocalConfig):
        """
        Initialize the local Whisper provider with configuration.
        
        Args:
            config: WhisperLocalConfig instance with provider configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, WhisperLocalConfig):
            raise ConfigurationError(
                f"Expected WhisperLocalConfig, got {type(config).__name__}"
            )
        
        self.config = config
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        """Load the Whisper model, downloading if necessary."""
        if self._model_loaded:
            return
            
        try:
            # Handle device parameter - whisper.load_model doesn't handle "auto" well
            device_param = self.config.device
            if device_param == "auto":
                # Let Whisper decide automatically by not specifying device
                self.model = whisper.load_model(self.config.model)
            else:
                # Use specific device
                self.model = whisper.load_model(self.config.model, device=device_param)
            self._model_loaded = True
        except Exception as e:
            raise ProviderError(
                f"Failed to load local Whisper model '{self.config.model}' "
                f"on device '{self.config.device}': {e}"
            )

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if not self._model_loaded:
            self._load_model()

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file using local Whisper.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (e.g., 'en', 'es')
            
        Returns:
            Raw transcript dictionary from local Whisper containing:
                - text: The transcribed text
                - segments: List of segment dictionaries with timing information
                - language: Detected or specified language
                
        Raises:
            AudioFileError: If audio file doesn't exist or format is not supported
            ProviderError: If transcription fails
            ProviderNotAvailableError: If model is not loaded
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
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if self.model is None:
            raise ProviderNotAvailableError(
                "Local Whisper model not loaded. Call validate_requirements() first."
            )
            
        try:
            # Transcribe with local model
            return self.model.transcribe(audio_path, language=language)
        except Exception as e:
            raise ProviderError(f"Local Whisper transcription failed: {e}")

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the provider name and model variant.
        
        Returns:
            Tuple of (provider_name, model_variant)
            
        Example:
            >>> provider = LocalWhisperProvider(config)
            >>> name, version = provider.get_engine_info()
            >>> print(f"{name} using {version}")
            "local-whisper using base"
        """
        return ("local-whisper", self.config.model)

    def validate_requirements(self) -> List[str]:
        """
        Validate that local Whisper is installed and model is available.
        
        This method checks:
        - OpenAI Whisper package is installed
        - Requested model is available
        - Device (CPU/CUDA) is available
        - Model can be loaded successfully
        
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Example:
            >>> provider = LocalWhisperProvider(config)
            >>> errors = provider.validate_requirements()
            >>> if errors:
            ...     print("Provider not ready:", errors)
            ... else:
            ...     print("Provider ready to use")
        """
        errors = []
        
        # Check if whisper package is available
        try:
            import whisper
        except ImportError:
            errors.append(
                "OpenAI Whisper package not installed. "
                "Install with: pip install openai-whisper"
            )
            return errors  # Can't continue without the package
        
        # Check available models
        try:
            available_models = whisper.available_models()
            if self.config.model not in available_models:
                errors.append(
                    f"Local Whisper model '{self.config.model}' not available. "
                    f"Available models: {', '.join(available_models)}"
                )
        except Exception as e:
            errors.append(f"Could not check available local Whisper models: {e}")
        
        # Check device availability
        if self.config.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(
                        "CUDA device requested but not available. "
                        "Use 'cpu' or 'auto' device instead."
                    )
            except ImportError:
                errors.append(
                    "PyTorch not installed. Required for local Whisper. "
                    "Install with: pip install torch"
                )
        
        # Try to load the model
        try:
            if not self._model_loaded:
                self._load_model()
        except Exception as e:
            errors.append(f"Failed to load local Whisper model '{self.config.model}': {e}")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats for local Whisper.
        
        Returns:
            List of supported file extensions
            
        Example:
            >>> provider = LocalWhisperProvider(config)
            >>> formats = provider.get_supported_formats()
            >>> print("Supported:", ", ".join(formats))
            "Supported: mp3, wav, flac, m4a, ogg, wma, aac"
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for local Whisper (always free).
        
        Args:
            audio_duration: Duration of audio in seconds (unused for local provider)
            
        Returns:
            None (local Whisper is always free)
            
        Example:
            >>> provider = LocalWhisperProvider(config)
            >>> cost = provider.estimate_cost(300.0)  # 5 minutes
            >>> print(cost)
            None
        """
        return None  # Local Whisper is always free
    
    def get_model_size_info(self) -> dict:
        """
        Get information about the current model size and requirements.
        
        This is a provider-specific method that provides additional information
        about the model being used, including parameter count, VRAM requirements,
        and relative speed.
        
        Returns:
            Dictionary with model size information:
                - model: Model name
                - device: Device being used
                - info: Dictionary with params, vram, and speed information
                
        Example:
            >>> provider = LocalWhisperProvider(config)
            >>> info = provider.get_model_size_info()
            >>> print(f"Model: {info['model']}, VRAM: {info['info']['vram']}")
            "Model: base, VRAM: ~1GB"
        """
        model_info = {
            'tiny': {'params': '39M', 'vram': '~1GB', 'speed': 'fastest'},
            'base': {'params': '74M', 'vram': '~1GB', 'speed': 'fast'},
            'small': {'params': '244M', 'vram': '~2GB', 'speed': 'medium'},
            'medium': {'params': '769M', 'vram': '~5GB', 'speed': 'slow'},
            'large': {'params': '1550M', 'vram': '~10GB', 'speed': 'slowest'}
        }
        
        return {
            'model': self.config.model,
            'device': self.config.device,
            'info': model_info.get(
                self.config.model,
                {'params': 'unknown', 'vram': 'unknown', 'speed': 'unknown'}
            )
        }
