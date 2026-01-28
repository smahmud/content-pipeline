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
                
                # Build client kwargs - only include credentials if explicitly provided
                # This allows boto3 to use its credential chain (env vars, AWS CLI config, IAM roles)
                transcribe_kwargs = {'region_name': self.region}
                s3_kwargs = {'region_name': self.region}
                
                if self.access_key_id and self.secret_access_key:
                    transcribe_kwargs['aws_access_key_id'] = self.access_key_id
                    transcribe_kwargs['aws_secret_access_key'] = self.secret_access_key
                    s3_kwargs['aws_access_key_id'] = self.access_key_id
                    s3_kwargs['aws_secret_access_key'] = self.secret_access_key
                
                self.client = boto3.client('transcribe', **transcribe_kwargs)
                self.s3_client = boto3.client('s3', **s3_kwargs)
            except ImportError:
                raise RuntimeError("boto3 package not installed. Install with: pip install boto3")
    
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
        return language_map.get(iso_code.lower(), self.language_code)
    
    def _get_s3_bucket_name(self) -> str:
        """Get or create S3 bucket name for transcription files."""
        # Use a bucket name from config or create a default one
        bucket_name = self.config.get('s3_bucket', f'content-pipeline-transcribe-{self.region}')
        return bucket_name
    
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
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
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
        import time
        import uuid
        
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
        
        # Convert ISO 639-1 language code to AWS language code if provided
        aws_language_code = self._convert_language_code(language) if language else self.language_code
        
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
                raise RuntimeError(f"AWS Transcribe job failed: {failure_reason}")
            
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
            'implementation_status': 'fully_implemented'
        }
    
    def get_file_size_limit(self) -> int:
        """
        Get the maximum file size limit in bytes.
        
        Returns:
            Maximum file size in bytes
        """
        return self.MAX_FILE_SIZE