"""
File: aws_transcribe.py

Implements the AWSTranscribeAdapter using AWS Transcribe service.
Conforms to the enhanced TranscriberAdapter protocol.

This adapter processes audio using AWS Transcribe cloud service.
"""
from typing import Optional, List, Tuple
import os
from pathlib import Path
from pipeline.utils.retry import retry
from pipeline.transcribers.adapters.base import TranscriberAdapter


class AWSTranscribeAdapter(TranscriberAdapter):
    """
    Transcribes audio using AWS Transcribe service.
    
    This adapter provides cloud-based transcription using AWS Transcribe.
    Enhanced in v0.6.5 to support requirement validation, format checking, and cost estimation.
    """
    
    # Supported audio formats by AWS Transcribe
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm']
    
    # AWS Transcribe pricing (as of 2024) - $0.024 per minute for standard
    COST_PER_MINUTE = 0.024
    
    # Maximum file size (2 GB)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB in bytes
    
    def __init__(self, access_key_id: Optional[str] = None, secret_access_key: Optional[str] = None,
                 region: str = "us-east-1", language_code: str = "en-US", **kwargs):
        """
        Initialize the AWS Transcribe adapter.
        
        Args:
            access_key_id: AWS access key ID (if None, will try to get from environment)
            secret_access_key: AWS secret access key (if None, will try to get from environment)
            region: AWS region for Transcribe service
            language_code: Language code for transcription (e.g., 'en-US', 'es-ES')
            **kwargs: Additional configuration options
        """
        self.access_key_id = access_key_id or self._get_access_key_from_env()
        self.secret_access_key = secret_access_key or self._get_secret_key_from_env()
        self.region = region
        self.language_code = language_code
        self.client = None
        
        # Store additional configuration
        self.config = kwargs

    def _get_access_key_from_env(self) -> Optional[str]:
        """Get AWS access key from environment variables."""
        return os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('CONTENT_PIPELINE_AWS_ACCESS_KEY_ID')

    def _get_secret_key_from_env(self) -> Optional[str]:
        """Get AWS secret key from environment variables."""
        return os.getenv('AWS_SECRET_ACCESS_KEY') or os.getenv('CONTENT_PIPELINE_AWS_SECRET_ACCESS_KEY')

    def _ensure_client_initialized(self):
        """Ensure the AWS Transcribe client is initialized."""
        if self.client is None:
            try:
                import boto3
                self.client = boto3.client(
                    'transcribe',
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region
                )
            except ImportError:
                raise RuntimeError("boto3 package not installed. Install with: pip install boto3")

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file using AWS Transcribe.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (ISO 639-1 code)
            
        Returns:
            Raw transcript dictionary from AWS Transcribe
            
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
            raise ValueError(f"File too large: {file_size / (1024*1024*1024):.1f}GB. Maximum size: {self.MAX_FILE_SIZE / (1024*1024*1024)}GB")
        
        # Ensure client is initialized
        self._ensure_client_initialized()
        
        if self.client is None:
            raise RuntimeError("AWS Transcribe client not initialized. Check credentials and validate_requirements().")
        
        # For now, return a placeholder implementation
        # TODO: Implement actual AWS Transcribe integration
        raise NotImplementedError(
            "AWS Transcribe adapter is not yet fully implemented. "
            "This is a placeholder for v0.6.5 architecture. "
            "Use --engine whisper-local or --engine whisper-api instead."
        )

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the engine name and model variant.
        
        Returns:
            Tuple of (engine_name, model_variant)
        """
        return ("aws-transcribe", f"{self.language_code}-{self.region}")

    def validate_requirements(self) -> List[str]:
        """
        Validate that AWS Transcribe is accessible and properly configured.
        
        Returns:
            List of error messages. Empty list means all requirements are met.
        """
        errors = []
        
        # Check if boto3 package is available
        try:
            import boto3
        except ImportError:
            errors.append("boto3 package not installed. Install with: pip install boto3")
            return errors  # Can't continue without the package
        
        # Check if AWS credentials are available
        if not self.access_key_id:
            errors.append("AWS access key not found. Set AWS_ACCESS_KEY_ID environment variable or provide access_key_id parameter.")
        
        if not self.secret_access_key:
            errors.append("AWS secret key not found. Set AWS_SECRET_ACCESS_KEY environment variable or provide secret_access_key parameter.")
        
        if not self.access_key_id or not self.secret_access_key:
            return errors  # Can't continue without credentials
        
        # Test AWS connectivity (optional - can be expensive)
        try:
            self._ensure_client_initialized()
            # We could test with a minimal API call here, but it would cost money
            # For now, just ensure client can be created
        except Exception as e:
            errors.append(f"Failed to initialize AWS Transcribe client: {e}")
        
        # Add implementation status warning
        errors.append("AWS Transcribe adapter is not yet fully implemented (placeholder for v0.6.5 architecture)")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats for AWS Transcribe.
        
        Returns:
            List of supported file extensions
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for AWS Transcribe.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD (or 0.0 if using credits)
        """
        if audio_duration <= 0:
            return 0.0
        
        # Convert seconds to minutes and calculate cost
        duration_minutes = audio_duration / 60.0
        
        # Note: If user has AWS credits, this would be $0
        # For now, return the standard pricing
        return round(duration_minutes * self.COST_PER_MINUTE, 4)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'language_code': self.language_code,
            'region': self.region,
            'max_file_size_gb': self.MAX_FILE_SIZE / (1024 * 1024 * 1024),
            'cost_per_minute_usd': self.COST_PER_MINUTE,
            'credentials_configured': bool(self.access_key_id and self.secret_access_key),
            'implementation_status': 'placeholder'
        }
    
    def get_file_size_limit(self) -> int:
        """
        Get the maximum file size limit in bytes.
        
        Returns:
            Maximum file size in bytes
        """
        return self.MAX_FILE_SIZE