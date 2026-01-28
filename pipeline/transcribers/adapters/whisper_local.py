"""
File: whisper_local.py

Implements the WhisperLocalAdapter using OpenAI's local Whisper model.
Conforms to the enhanced TranscriberAdapter protocol.

This adapter processes audio entirely on the local machine for privacy-focused transcription.
"""
from typing import Optional, List, Tuple
import whisper
import os
from pathlib import Path
from pipeline.utils.retry import retry
from pipeline.transcribers.adapters.base import TranscriberAdapter


class WhisperLocalAdapter(TranscriberAdapter):
    """
    Transcribes audio using a locally loaded Whisper model.
    
    This adapter ensures complete privacy by processing audio entirely on the local machine.
    Enhanced in v0.6.5 to support requirement validation, format checking, and improved error handling.
    """
    
    # Supported audio formats by local Whisper
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'wma', 'aac']
    
    def __init__(self, model_name: str = "base", device: str = "cpu", **kwargs):
        """
        Initialize the local Whisper adapter with the specified model variant.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ('cpu', 'cuda', 'auto')
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._model_loaded = False
        
        # Store additional configuration
        self.config = kwargs

    def _load_model(self):
        """Load the Whisper model, downloading if necessary."""
        if self._model_loaded:
            return
            
        try:
            # Handle device parameter - whisper.load_model doesn't handle "auto" well
            device_param = self.device
            if device_param == "auto":
                # Let Whisper decide automatically by not specifying device
                self.model = whisper.load_model(self.model_name)
            else:
                # Use specific device
                self.model = whisper.load_model(self.model_name, device=device_param)
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load local Whisper model '{self.model_name}' on device '{self.device}': {e}")

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
            language: Optional language hint for transcription
            
        Returns:
            Raw transcript dictionary from local Whisper
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check if file format is supported
        file_ext = Path(audio_path).suffix.lower().lstrip('.')
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if self.model is None:
            raise RuntimeError("Local Whisper model not loaded. Call validate_requirements() first.")
            
        try:
            # Transcribe with local model
            return self.model.transcribe(audio_path, language=language)
        except Exception as e:
            raise RuntimeError(f"Local Whisper transcription failed: {e}")

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the engine name and model variant.
        
        Returns:
            Tuple of (engine_name, model_variant)
        """
        return ("whisper-local", self.model_name)

    def validate_requirements(self) -> List[str]:
        """
        Validate that local Whisper is installed and model is available.
        
        Returns:
            List of error messages. Empty list means all requirements are met.
        """
        errors = []
        
        # Check if whisper package is available
        try:
            import whisper
        except ImportError:
            errors.append("OpenAI Whisper package not installed. Install with: pip install openai-whisper")
            return errors  # Can't continue without the package
        
        # Check available models
        try:
            available_models = whisper.available_models()
            if self.model_name not in available_models:
                errors.append(f"Local Whisper model '{self.model_name}' not available. Available models: {list(available_models)}")
        except Exception as e:
            errors.append(f"Could not check available local Whisper models: {e}")
        
        # Check device availability
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("CUDA device requested but not available. Use 'cpu' or 'auto' device instead.")
            except ImportError:
                errors.append("PyTorch not installed. Required for local Whisper. Install with: pip install torch")
        
        # Try to load the model
        try:
            if not self._model_loaded:
                self._load_model()
        except Exception as e:
            errors.append(f"Failed to load local Whisper model '{self.model_name}': {e}")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats for local Whisper.
        
        Returns:
            List of supported file extensions
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for local Whisper (always free).
        
        Args:
            audio_duration: Duration of audio in seconds (unused for local)
            
        Returns:
            None (local Whisper is always free)
        """
        return None  # Local Whisper is always free
    
    def get_model_size_info(self) -> dict:
        """
        Get information about the current model size and requirements.
        
        Returns:
            Dictionary with model size information
        """
        model_info = {
            'tiny': {'params': '39M', 'vram': '~1GB', 'speed': 'fastest'},
            'base': {'params': '74M', 'vram': '~1GB', 'speed': 'fast'},
            'small': {'params': '244M', 'vram': '~2GB', 'speed': 'medium'},
            'medium': {'params': '769M', 'vram': '~5GB', 'speed': 'slow'},
            'large': {'params': '1550M', 'vram': '~10GB', 'speed': 'slowest'}
        }
        
        return {
            'model': self.model_name,
            'device': self.device,
            'info': model_info.get(self.model_name, {'params': 'unknown', 'vram': 'unknown', 'speed': 'unknown'})
        }