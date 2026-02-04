"""
Cloud AWS Transcribe Transcription Provider

Implements the CloudAWSTranscribeProvider using AWS Transcribe service.
Conforms to the TranscriberProvider protocol.

This provider processes audio using AWS Transcribe cloud service with automatic S3 integration.
Migrated from pipeline.transcribers.adapters.aws_transcribe as part of the infrastructure
refactoring to establish clean separation between infrastructure and domain layers.

**Validates: Requirements 2.2, 2.3**
"""
from typing import Optional, List, Tuple
import os
from pathlib import Path

from pipeline.utils.retry import retry
from pipeline.transcription.providers.base import TranscriberProvider
from pipeline.transcription.config import AWSTranscribeConfig
from pipeline.transcription.errors import (
    AudioFileError,
    ProviderError,
    ProviderNotAvailableError,
    ConfigurationError
)


class CloudAWSTranscribeProvider(TranscriberProvider):
    """
    Transcribes audio using AWS Transcribe service.
    
    This provider provides cloud-based transcription using AWS Transcribe with automatic
    S3 file management. Supports requirement validation, format checking, and cost estimation.
    
    Configuration is provided via AWSTranscribeConfig, which supports:
    - AWS credentials (access key ID, secret access key)
    - Region selection
    - Language code configuration
    - S3 bucket configuration
    - Timeout and retry configuration
    
    Example:
        >>> from pipeline.transcription.config import AWSTranscribeConfig
        >>> config = AWSTranscribeConfig(
        ...     access_key_id="AKIA...",
        ...     secret_access_key="...",
        ...     region="us-east-1",
        ...     language_code="en-US"
        ... )
        >>> provider = CloudAWSTranscribeProvider(config)
        >>> result = provider.transcribe("audio.mp3")
        >>> print(result['results']['transcripts'][0]['transcript'])
    """
    
    # Supported audio formats by AWS Transcribe
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm']
    
    # AWS Transcribe pricing (as of 2024) - $0.024 per minute for standard
    COST_PER_MINUTE = 0.024
    
    # Maximum file size (2 GB)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB in bytes
    
    def __init__(self, config: AWSTranscribeConfig):
        """
        Initialize the AWS Transcribe provider with configuration.
        
        Args:
            config: AWSTranscribeConfig instance with provider configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, AWSTranscribeConfig):
            raise ConfigurationError(
                f"Expected AWSTranscribeConfig, got {type(config).__name__}"
            )
        
        self.config = config
        self.client = None
        self.s3_client = None

    def _ensure_client_initialized(self):
        """Ensure the AWS Transcribe client is initialized."""
        if self.client is None:
            try:
                import boto3
                
                # Build client kwargs - only include credentials if explicitly provided
                # This allows boto3 to use its credential chain (env vars, AWS CLI config, IAM roles)
                transcribe_kwargs = {'region_name': self.config.region}
                s3_kwargs = {'region_name': self.config.region}
                
                if self.config.access_key_id and self.config.secret_access_key:
                    transcribe_kwargs['aws_access_key_id'] = self.config.access_key_id
                    transcribe_kwargs['aws_secret_access_key'] = self.config.secret_access_key
                    s3_kwargs['aws_access_key_id'] = self.config.access_key_id
                    s3_kwargs['aws_secret_access_key'] = self.config.secret_access_key
                
                self.client = boto3.client('transcribe', **transcribe_kwargs)
                self.s3_client = boto3.client('s3', **s3_kwargs)
            except ImportError:
                raise ProviderNotAvailableError(
                    "boto3 package not installed. Install with: pip install boto3"
                )
    
    def _convert_language_code(self, iso_code: str) -> str:
        """
        Convert ISO 639-1 language code to AWS Transcribe language code.
        
        Args:
            iso_code: ISO 639-1 language code (e.g., 'en', 'es')
            
        Returns:
            AWS Transcribe language code (e.g., 'en-US', 'es-ES')
        """
        # Map common ISO codes to AWS codes
        language_map = {
            'en': 'en-US',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'hi': 'hi-IN',
            'ru': 'ru-RU',
        }
        return language_map.get(iso_code.lower(), self.config.language_code)
    
    def _get_s3_bucket_name(self) -> str:
        """Get or create S3 bucket name for transcription files."""
        # Use configured bucket name or create a default one
        if self.config.s3_bucket:
            return self.config.s3_bucket
        return f'content-pipeline-transcribe-{self.config.region}'
    
    def _upload_to_s3(self, audio_path: str, job_name: str) -> str:
        """
        Upload audio file to S3 and return the S3 URI.
        
        Args:
            audio_path: Local path to audio file
            job_name: Unique job name for the file
            
        Returns:
            S3 URI of the uploaded file
        """
        bucket_name = self._get_s3_bucket_name()
        file_ext = Path(audio_path).suffix
        s3_key = f"transcribe-input/{job_name}{file_ext}"
        
        # Ensure bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except:
            # Create bucket if it doesn't exist
            if self.config.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.config.region}
                )
        
        # Upload file
        self.s3_client.upload_file(audio_path, bucket_name, s3_key)
        
        return f"s3://{bucket_name}/{s3_key}"
    
    def _delete_from_s3(self, s3_uri: str):
        """Delete file from S3."""
        # Parse S3 URI
        parts = s3_uri.replace('s3://', '').split('/', 1)
        if len(parts) == 2:
            bucket_name, s3_key = parts
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
    
    def _download_transcript(self, transcript_uri: str) -> dict:
        """
        Download and parse transcript from AWS Transcribe.
        
        Args:
            transcript_uri: URI to the transcript JSON file
            
        Returns:
            Parsed transcript dictionary
        """
        import json
        import urllib.request
        
        # Download transcript JSON
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_data = json.loads(response.read().decode('utf-8'))
        
        return transcript_data

    @retry(max_attempts=3)
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Run transcription on the given audio file using AWS Transcribe.
        
        This method uploads the audio file to S3, starts a transcription job,
        waits for completion, downloads the transcript, and cleans up resources.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (ISO 639-1 code, e.g., 'en', 'es')
            
        Returns:
            Raw transcript dictionary from AWS Transcribe containing:
                - results: Transcription results with transcripts and items
                - status: Job status
                
        Raises:
            AudioFileError: If audio file doesn't exist, format is not supported, or file is too large
            ProviderError: If transcription fails
            ProviderNotAvailableError: If client is not initialized
        """
        import time
        import uuid
        
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
                f"File too large: {file_size / (1024*1024*1024):.1f}GB. "
                f"Maximum size: {self.MAX_FILE_SIZE / (1024*1024*1024)}GB"
            )
        
        # Ensure client is initialized
        self._ensure_client_initialized()
        
        if self.client is None:
            raise ProviderNotAvailableError(
                "AWS Transcribe client not initialized. Check credentials and validate_requirements()."
            )
        
        # Convert ISO 639-1 language code to AWS language code if provided
        aws_language_code = self._convert_language_code(language) if language else self.config.language_code
        
        # Generate unique job name
        job_name = f"content-pipeline-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Upload file to S3 (AWS Transcribe requires S3 URI)
        s3_uri = self._upload_to_s3(audio_path, job_name)
        
        try:
            # Start transcription job
            self.client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                MediaFormat=file_ext,
                LanguageCode=aws_language_code
            )
            
            # Wait for job to complete
            while True:
                status = self.client.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                    
                time.sleep(2)  # Poll every 2 seconds
            
            if job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise ProviderError(f"AWS Transcribe job failed: {failure_reason}")
            
            # Get transcript
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript_data = self._download_transcript(transcript_uri)
            
            return transcript_data
            
        finally:
            # Clean up: delete transcription job
            try:
                self.client.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception:
                pass  # Ignore cleanup errors
            
            # Clean up: delete S3 file
            try:
                self._delete_from_s3(s3_uri)
            except Exception:
                pass  # Ignore cleanup errors

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the provider name and configuration variant.
        
        Returns:
            Tuple of (provider_name, configuration_variant)
            
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> name, version = provider.get_engine_info()
            >>> print(f"{name} using {version}")
            "cloud-aws-transcribe using en-US-us-east-1"
        """
        return ("cloud-aws-transcribe", f"{self.config.language_code}-{self.config.region}")

    def validate_requirements(self) -> List[str]:
        """
        Validate that AWS Transcribe is accessible and properly configured.
        
        This method checks:
        - boto3 package is installed
        - AWS credentials are available (explicit, env vars, AWS CLI, or IAM roles)
        - Credentials are valid and can access AWS Transcribe
        
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> errors = provider.validate_requirements()
            >>> if errors:
            ...     print("Provider not ready:", errors)
            ... else:
            ...     print("Provider ready to use")
        """
        errors = []
        
        # Check if boto3 package is available
        try:
            import boto3
        except ImportError:
            errors.append("boto3 package not installed. Install with: pip install boto3")
            return errors  # Can't continue without the package
        
        # Try to initialize client - boto3 will use its credential chain
        # (explicit credentials, env vars, AWS CLI config, IAM roles)
        try:
            self._ensure_client_initialized()
            
            # Test that we can actually use the credentials by making a simple API call
            # This validates that credentials exist and are valid
            try:
                # Try to list transcription jobs (doesn't cost anything)
                self.client.list_transcription_jobs(MaxResults=1)
            except Exception as e:
                error_msg = str(e)
                if 'credentials' in error_msg.lower() or 'access' in error_msg.lower():
                    errors.append(
                        "AWS credentials not found or invalid. Provide credentials using one of:\n"
                        "  1. AWS CLI: Run 'aws configure' to set up credentials\n"
                        "  2. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                        "  3. Configuration file: Set access_key_id and secret_access_key in config.yaml"
                    )
                else:
                    errors.append(f"Failed to connect to AWS Transcribe: {e}")
        except Exception as e:
            errors.append(f"Failed to initialize AWS Transcribe client: {e}")
        
        return errors

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats for AWS Transcribe.
        
        Returns:
            List of supported file extensions
            
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> formats = provider.get_supported_formats()
            >>> print("Supported:", ", ".join(formats))
            "Supported: mp3, mp4, wav, flac, ogg, amr, webm"
        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost for AWS Transcribe.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD (or 0.0 if using AWS credits)
            
        Note:
            Cost is calculated using the configured cost_per_minute_usd rate.
            Default rate is $0.024/minute but can be overridden in configuration
            for enterprise customers with custom pricing or regional differences.
            If you have AWS credits, the actual cost may be $0.
            
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> cost = provider.estimate_cost(300.0)  # 5 minutes
            >>> print(f"Estimated cost: ${cost:.4f}")
            "Estimated cost: $0.1200"
        """
        if audio_duration <= 0:
            return 0.0
        
        # Convert seconds to minutes and calculate cost using config value
        duration_minutes = audio_duration / 60.0
        
        # Note: If user has AWS credits, this would be $0
        # For now, return the configured pricing
        return round(duration_minutes * self.config.cost_per_minute_usd, 4)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current configuration.
        
        This is a provider-specific method that provides additional information
        about the provider configuration.
        
        Returns:
            Dictionary with configuration information:
                - language_code: AWS language code
                - region: AWS region
                - max_file_size_gb: Maximum file size in GB
                - cost_per_minute_usd: Cost per minute in USD
                - credentials_configured: Whether explicit credentials are configured
                - implementation_status: Implementation status
                
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> info = provider.get_model_info()
            >>> print(f"Region: {info['region']}, Cost: ${info['cost_per_minute_usd']}/min")
            "Region: us-east-1, Cost: $0.024/min"
        """
        return {
            'language_code': self.config.language_code,
            'region': self.config.region,
            'max_file_size_gb': self.MAX_FILE_SIZE / (1024 * 1024 * 1024),
            'cost_per_minute_usd': self.config.cost_per_minute_usd,
            'credentials_configured': bool(self.config.access_key_id and self.config.secret_access_key),
            'implementation_status': 'fully_implemented'
        }
    
    def get_file_size_limit(self) -> int:
        """
        Get the maximum file size limit in bytes.
        
        This is a provider-specific method that returns the maximum file size
        that can be processed by AWS Transcribe.
        
        Returns:
            Maximum file size in bytes
            
        Example:
            >>> provider = CloudAWSTranscribeProvider(config)
            >>> limit = provider.get_file_size_limit()
            >>> print(f"Max file size: {limit / (1024*1024*1024)}GB")
            "Max file size: 2.0GB"
        """
        return self.MAX_FILE_SIZE
