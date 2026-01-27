"""
File: whisper_api.py

Implements the WhisperAPIAdapter using OpenAI's Whisper API.
Conforms to the enhanced TranscriberAdapter protocol.

This adapter processes audio using OpenAI's cloud-based Whisper API for high-quality transcription.
"""
from typing import Optional, List, Tuple
import os
from pathlib import Path
from pipeline.utils.retry import retry
from pipeline.transcribers.adapters.base import TranscriberAdapter


class WhisperAPIAdapter(TranscriberAdapter):
    """
    Transcribes audio using OpenAI's Whisper API.
    
    This adapter provides high-quality transcription using OpenAI's cloud-based service.
    Enhanced in v0.6.5 to support requirement validation, format checking, and cost estimation.
    """
    
    # Supported audio formats by OpenAI Whisper API
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
    
    # API pricing (as of 2024) - $0.006 per minute
    COST_PER_MINUTE = 0.006
    
    # Maximum file size (25 MB)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes
    
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1", 
                 temperature: float = 0.0, response_format: str = "json", **kwargs):
        """
        Initialize the OpenAI Whisper API adapter.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Whisper model to use (whisper-1, gpt-4o-transcribe, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            response_format: Response format (json, text, srt, verbose_json, vtt)
            **kwargs: Additional configuration options
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.client = None
        
        # Store additional configuration
        self.config = kwargs

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get OpenAI API key from environment variables."""
        return os.getenv('OPENAI_API_KEY') or os.getenv('CONTENT_PIPELINE_OPENAI_API_KEY')

    def _ensure_client_initialized(self):
        """Ensure the OpenAI client is initialized."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("OpenAI package not installed. Install with: pip install openai")

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file using OpenAI Whisper API.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (ISO 639-1 code)
            
        Returns:
            Raw transcript dictionary from OpenAI Whisper API
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported or file is too large
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check if file format is supported
        file_ext = Path(audio_path).suffix.lower().lstrip('.')
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB. Maximum size: {self.MAX_FILE_SIZE / (1024*1024)}MB")
        
        # Ensure client is initialized
        self._ensure_client_initialized()
        
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key and validate_requirements().")
            
        try:
            # Prepare transcription parameters
            transcription_params = {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": self.response_format
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
            if self.response_format == "json":
                return {
                    "text": response.text,
                    "segments": getattr(response, 'segments', []),
                    "language": getattr(response, 'language', language or 'unknown')
                }
            elif self.response_format == "verbose_json":
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
            raise RuntimeError(f"OpenAI Whisper API transcription failed: {e}")

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the engine name and model variant.
        
        Returns:
            Tuple of (engine_name, model_variant)
        """
        return ("whisper-api", self.model)

    def validate_requirements(self) -> List[str]:
        """
        Validate that OpenAI API is accessible and properly configured.
        
        Returns:
            List of error messages. Empty list means all requirements are met.
        """
        errors = []
        
        # Check if openai package is available
        try:
            import openai
        except ImportError:
            errors.append("OpenAI package not installed. Install with: pip install openai")
            return errors  # Can't continue without the package
        
        # Check if API key is available
        if not self.api_key:
            errors.append("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            return errors  # Can't continue without API key
        
        # Validate API key format (should start with 'sk-')
        if not self.api_key.startswith('sk-'):
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
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for OpenAI Whisper API.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD
        """
        if audio_duration <= 0:
            return 0.0
        
        # Convert seconds to minutes and calculate cost
        duration_minutes = audio_duration / 60.0
        return round(duration_minutes * self.COST_PER_MINUTE, 4)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model': self.model,
            'temperature': self.temperature,
            'response_format': self.response_format,
            'max_file_size_mb': self.MAX_FILE_SIZE / (1024 * 1024),
            'cost_per_minute_usd': self.COST_PER_MINUTE,
            'api_key_configured': bool(self.api_key)
        }
    
    def get_file_size_limit(self) -> int:
        """
        Get the maximum file size limit in bytes.
        
        Returns:
            Maximum file size in bytes
        """
        return self.MAX_FILE_SIZE