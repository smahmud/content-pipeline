"""
File: whisper.py

Implements the WhisperAdapter using OpenAI's Whisper model.
Conforms to the enhanced TranscriberAdapter protocol.
"""
from typing import Optional, List, Tuple
import whisper
import os
from pathlib import Path
from pipeline.utils.retry import retry
from pipeline.transcribers.adapters.base import TranscriberAdapter


class WhisperAdapter(TranscriberAdapter):
    """
    Transcribes audio using a locally loaded Whisper model.
    Enhanced in v0.6.5 to support requirement validation and format checking.
    """
    
    # Supported audio formats by Whisper
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'wma', 'aac']
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the Whisper adapter with the specified model variant.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        """Load the Whisper model, downloading if necessary."""
        if self._model_loaded:
            return
            
        try:
            # type: ignore[attr-defined]
            self.model = whisper.load_model(self.model_name) # type: ignore[attr-defined]
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}': {e}")

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if not self._model_loaded:
            self._load_model()

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription
            
        Returns:
            Raw transcript dictionary from Whisper
            
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
            raise RuntimeError("Whisper model not loaded. Call validate_requirements() first.")
            
        try:
            return self.model.transcribe(audio_path, language=language)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the engine name and model variant.
        
        Returns:
            Tuple of (engine_name, model_variant)
        """
        return ("whisper", self.model_name)

    def validate_requirements(self) -> List[str]:
        """
        Validate that Whisper is installed and model is available.
        
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
                errors.append(f"Model '{self.model_name}' not available. Available models: {list(available_models)}")
        except Exception as e:
            errors.append(f"Could not check available models: {e}")
        
        # Try to load the model
        try:
            if not self._model_loaded:
                self._load_model()
        except Exception as e:
            errors.append(f"Failed to load Whisper model '{self.model_name}': {e}")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats.
        
        Returns:
            List of supported file extensions
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for local Whisper (free).
        
        Args:
            audio_duration: Duration of audio in seconds (unused for local)
            
        Returns:
            None (local Whisper is free)
        """
        return None  # Local Whisper is free
