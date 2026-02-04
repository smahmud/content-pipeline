"""
Unit Tests: CloudAWSTranscribeProvider

Tests for the Cloud AWS Transcribe transcription provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Transcribe method with S3 integration
- Estimate cost method with accurate pricing
- Validate requirements method
- Error handling for missing files, unsupported formats, and file size limits
- Get supported formats method
- Get engine info method
- Model info method
- S3 upload/download/cleanup operations
- Language code conversion

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
from pathlib import Path

from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider
from pipeline.transcription.config import AWSTranscribeConfig
from pipeline.transcription.errors import (
    AudioFileError,
    ProviderError,
    ProviderNotAvailableError,
    ConfigurationError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def aws_config():
    """Create a test AWS Transcribe configuration."""
    return AWSTranscribeConfig(
        access_key_id="AKIATEST12345",
        secret_access_key="test_secret_key",
        region="us-east-1",
        language_code="en-US",
        s3_bucket="test-bucket",
        timeout=600,
        retry_attempts=3,
        retry_delay=2.0
    )


@pytest.fixture
def aws_provider(aws_config):
    """Create a CloudAWSTranscribeProvider instance."""
    return CloudAWSTranscribeProvider(aws_config)


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a mock audio file."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_text("fake audio content")
    return str(audio_file)


# ============================================================================
# Test: Initialization
# ============================================================================

def test_initialization_with_valid_config(aws_config):
    """Test provider initialization with valid configuration."""
    provider = CloudAWSTranscribeProvider(aws_config)
    
    assert provider is not None
    assert provider.config == aws_config
    assert provider.config.access_key_id == "AKIATEST12345"
    assert provider.config.region == "us-east-1"
    assert provider.client is None  # Client not initialized yet
    assert provider.s3_client is None  # S3 client not initialized yet


def test_initialization_with_invalid_config():
    """Test provider initialization with invalid configuration type."""
    with pytest.raises(ConfigurationError) as exc_info:
        CloudAWSTranscribeProvider("not a config object")
    
    assert "Expected AWSTranscribeConfig" in str(exc_info.value)


def test_initialization_stores_config(aws_config):
    """Test that provider stores configuration object."""
    provider = CloudAWSTranscribeProvider(aws_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, AWSTranscribeConfig)


# ============================================================================
# Test: Client Initialization
# ============================================================================

@patch('boto3.client')
def test_ensure_client_initialized_with_credentials(mock_boto_client, aws_provider):
    """Test client initialization with explicit credentials."""
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    
    aws_provider._ensure_client_initialized()
    
    assert aws_provider.client is not None
    assert aws_provider.s3_client is not None
    
    # Verify boto3.client was called with correct parameters
    assert mock_boto_client.call_count == 2
    
    # Check transcribe client call
    transcribe_call = mock_boto_client.call_args_list[0]
    assert transcribe_call[0][0] == 'transcribe'
    assert transcribe_call[1]['region_name'] == 'us-east-1'
    assert transcribe_call[1]['aws_access_key_id'] == 'AKIATEST12345'
    
    # Check S3 client call
    s3_call = mock_boto_client.call_args_list[1]
    assert s3_call[0][0] == 's3'


@patch('boto3.client')
def test_ensure_client_initialized_without_credentials(mock_boto_client, aws_config):
    """Test client initialization without explicit credentials (uses AWS credential chain)."""
    aws_config.access_key_id = None
    aws_config.secret_access_key = None
    provider = CloudAWSTranscribeProvider(aws_config)
    
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    
    provider._ensure_client_initialized()
    
    # Verify boto3.client was called without explicit credentials
    transcribe_call = mock_boto_client.call_args_list[0]
    assert 'aws_access_key_id' not in transcribe_call[1]
    assert 'aws_secret_access_key' not in transcribe_call[1]


def test_ensure_client_initialized_without_boto3(aws_provider):
    """Test client initialization when boto3 is not installed."""
    with patch('boto3.client', side_effect=ImportError):
        with pytest.raises(ProviderNotAvailableError) as exc_info:
            aws_provider._ensure_client_initialized()
        
        assert "not installed" in str(exc_info.value)


# ============================================================================
# Test: Language Code Conversion
# ============================================================================

def test_convert_language_code_common_languages(aws_provider):
    """Test language code conversion for common languages."""
    assert aws_provider._convert_language_code('en') == 'en-US'
    assert aws_provider._convert_language_code('es') == 'es-ES'
    assert aws_provider._convert_language_code('fr') == 'fr-FR'
    assert aws_provider._convert_language_code('de') == 'de-DE'
    assert aws_provider._convert_language_code('ja') == 'ja-JP'


def test_convert_language_code_unknown_language(aws_provider):
    """Test language code conversion for unknown language."""
    # Should return the configured default language code
    result = aws_provider._convert_language_code('xyz')
    assert result == 'en-US'  # Default from config


def test_convert_language_code_case_insensitive(aws_provider):
    """Test language code conversion is case-insensitive."""
    assert aws_provider._convert_language_code('EN') == 'en-US'
    assert aws_provider._convert_language_code('Es') == 'es-ES'


# ============================================================================
# Test: S3 Operations
# ============================================================================

def test_get_s3_bucket_name_with_configured_bucket(aws_provider):
    """Test S3 bucket name retrieval with configured bucket."""
    bucket_name = aws_provider._get_s3_bucket_name()
    
    assert bucket_name == "test-bucket"


def test_get_s3_bucket_name_without_configured_bucket(aws_config):
    """Test S3 bucket name retrieval without configured bucket."""
    aws_config.s3_bucket = None
    provider = CloudAWSTranscribeProvider(aws_config)
    
    bucket_name = provider._get_s3_bucket_name()
    
    # Should generate default bucket name
    assert bucket_name == f'content-pipeline-transcribe-{aws_config.region}'


@patch('boto3.client')
def test_upload_to_s3_existing_bucket(mock_boto_client, aws_provider, mock_audio_file):
    """Test uploading file to existing S3 bucket."""
    mock_s3_client = Mock()
    mock_boto_client.return_value = mock_s3_client
    aws_provider.s3_client = mock_s3_client
    
    # Mock bucket exists
    mock_s3_client.head_bucket.return_value = {}
    
    s3_uri = aws_provider._upload_to_s3(mock_audio_file, "test-job")
    
    # Verify upload was called
    mock_s3_client.upload_file.assert_called_once()
    
    # Verify S3 URI format
    assert s3_uri.startswith("s3://test-bucket/transcribe-input/test-job")
    assert s3_uri.endswith(".mp3")


@patch('boto3.client')
def test_upload_to_s3_creates_bucket_if_not_exists(mock_boto_client, aws_provider, mock_audio_file):
    """Test uploading file creates bucket if it doesn't exist."""
    mock_s3_client = Mock()
    mock_boto_client.return_value = mock_s3_client
    aws_provider.s3_client = mock_s3_client
    
    # Mock bucket doesn't exist
    mock_s3_client.head_bucket.side_effect = Exception("Bucket not found")
    
    s3_uri = aws_provider._upload_to_s3(mock_audio_file, "test-job")
    
    # Verify bucket creation was attempted
    mock_s3_client.create_bucket.assert_called_once()


@patch('boto3.client')
def test_delete_from_s3(mock_boto_client, aws_provider):
    """Test deleting file from S3."""
    mock_s3_client = Mock()
    mock_boto_client.return_value = mock_s3_client
    aws_provider.s3_client = mock_s3_client
    
    s3_uri = "s3://test-bucket/transcribe-input/test-job.mp3"
    aws_provider._delete_from_s3(s3_uri)
    
    # Verify delete was called
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="transcribe-input/test-job.mp3"
    )


# ============================================================================
# Test: Transcribe Method
# ============================================================================

@patch('time.sleep')
@patch('urllib.request.urlopen')
@patch('boto3.client')
def test_transcribe_with_valid_file(mock_boto_client, mock_urlopen, mock_sleep, aws_provider, mock_audio_file):
    """Test transcribe method with valid audio file."""
    # Mock AWS clients
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    aws_provider.client = mock_transcribe_client
    aws_provider.s3_client = mock_s3_client
    
    # Mock S3 operations
    mock_s3_client.head_bucket.return_value = {}
    
    # Mock transcription job
    mock_transcribe_client.start_transcription_job.return_value = {}
    mock_transcribe_client.get_transcription_job.return_value = {
        'TranscriptionJob': {
            'TranscriptionJobStatus': 'COMPLETED',
            'Transcript': {
                'TranscriptFileUri': 'https://s3.amazonaws.com/bucket/transcript.json'
            }
        }
    }
    
    # Mock transcript download
    mock_response = Mock()
    mock_response.read.return_value = b'{"results": {"transcripts": [{"transcript": "Test transcription"}]}}'
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    mock_urlopen.return_value = mock_response
    
    # Transcribe
    result = aws_provider.transcribe(mock_audio_file, language="en")
    
    # Verify result
    assert "results" in result
    assert result["results"]["transcripts"][0]["transcript"] == "Test transcription"
    
    # Verify job was started
    mock_transcribe_client.start_transcription_job.assert_called_once()
    
    # Verify cleanup was attempted
    mock_transcribe_client.delete_transcription_job.assert_called_once()


def test_transcribe_with_nonexistent_file(aws_provider):
    """Test transcribe method with nonexistent file."""
    with pytest.raises(AudioFileError) as exc_info:
        aws_provider.transcribe("/nonexistent/file.mp3")
    
    assert "not found" in str(exc_info.value)


def test_transcribe_with_unsupported_format(aws_provider, tmp_path):
    """Test transcribe method with unsupported audio format."""
    # Create file with unsupported extension
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("fake content")
    
    with pytest.raises(AudioFileError) as exc_info:
        aws_provider.transcribe(str(unsupported_file))
    
    assert "Unsupported audio format" in str(exc_info.value)


def test_transcribe_with_file_too_large(aws_provider, tmp_path):
    """Test transcribe method with file exceeding size limit."""
    # Create a file that reports as too large
    large_file = tmp_path / "large_audio.mp3"
    large_file.write_text("x" * 100)
    
    with patch('os.path.getsize', return_value=3 * 1024 * 1024 * 1024):  # 3 GB
        with pytest.raises(AudioFileError) as exc_info:
            aws_provider.transcribe(str(large_file))
        
        assert "too large" in str(exc_info.value).lower()


@patch('time.sleep')
@patch('boto3.client')
def test_transcribe_job_failure(mock_boto_client, mock_sleep, aws_provider, mock_audio_file):
    """Test transcribe method when transcription job fails."""
    # Mock AWS clients
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    aws_provider.client = mock_transcribe_client
    aws_provider.s3_client = mock_s3_client
    
    # Mock S3 operations
    mock_s3_client.head_bucket.return_value = {}
    
    # Mock transcription job failure
    mock_transcribe_client.start_transcription_job.return_value = {}
    mock_transcribe_client.get_transcription_job.return_value = {
        'TranscriptionJob': {
            'TranscriptionJobStatus': 'FAILED',
            'FailureReason': 'Audio quality too low'
        }
    }
    
    with pytest.raises(ProviderError) as exc_info:
        aws_provider.transcribe(mock_audio_file)
    
    assert "failed" in str(exc_info.value).lower()
    assert "Audio quality too low" in str(exc_info.value)


# ============================================================================
# Test: Estimate Cost
# ============================================================================

def test_estimate_cost_calculates_correctly(aws_provider):
    """Test estimate_cost calculates cost based on duration."""
    # 5 minutes = 300 seconds
    cost = aws_provider.estimate_cost(300.0)
    
    # Cost should be 5 * $0.024 = $0.12
    assert cost == 0.12


def test_estimate_cost_with_zero_duration(aws_provider):
    """Test estimate_cost with zero duration."""
    cost = aws_provider.estimate_cost(0.0)
    
    assert cost == 0.0


def test_estimate_cost_with_negative_duration(aws_provider):
    """Test estimate_cost with negative duration."""
    cost = aws_provider.estimate_cost(-10.0)
    
    assert cost == 0.0


def test_estimate_cost_rounds_correctly(aws_provider):
    """Test estimate_cost rounds to 4 decimal places."""
    # 123 seconds = 2.05 minutes = $0.0492
    cost = aws_provider.estimate_cost(123.0)
    
    assert isinstance(cost, float)
    assert len(str(cost).split('.')[-1]) <= 4  # At most 4 decimal places


# ============================================================================
# Test: Validate Requirements
# ============================================================================

@patch('boto3.client')
def test_validate_requirements_with_valid_credentials(mock_boto_client, aws_provider):
    """Test validate_requirements with valid credentials."""
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    
    # Mock successful API call
    mock_transcribe_client.list_transcription_jobs.return_value = {'TranscriptionJobSummaries': []}
    
    errors = aws_provider.validate_requirements()
    
    assert len(errors) == 0


@patch('boto3.client')
def test_validate_requirements_with_invalid_credentials(mock_boto_client, aws_provider):
    """Test validate_requirements with invalid credentials."""
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    
    # Mock authentication error
    mock_transcribe_client.list_transcription_jobs.side_effect = Exception("Invalid credentials")
    
    errors = aws_provider.validate_requirements()
    
    assert len(errors) > 0
    assert any("credentials" in error.lower() or "invalid" in error.lower() for error in errors)


def test_validate_requirements_without_boto3(aws_provider):
    """Test validate_requirements when boto3 is not installed."""
    with patch.dict('sys.modules', {'boto3': None}):
        errors = aws_provider.validate_requirements()
        
        assert len(errors) > 0
        assert any("not installed" in error for error in errors)


# ============================================================================
# Test: Get Supported Formats
# ============================================================================

def test_get_supported_formats(aws_provider):
    """Test get_supported_formats returns correct formats."""
    formats = aws_provider.get_supported_formats()
    
    assert isinstance(formats, list)
    assert len(formats) > 0
    assert 'mp3' in formats
    assert 'mp4' in formats
    assert 'wav' in formats
    assert 'flac' in formats


def test_get_supported_formats_returns_copy(aws_provider):
    """Test get_supported_formats returns a copy, not the original list."""
    formats1 = aws_provider.get_supported_formats()
    formats2 = aws_provider.get_supported_formats()
    
    # Modify one list
    formats1.append('xyz')
    
    # Verify the other list is not affected
    assert 'xyz' not in formats2


# ============================================================================
# Test: Get Engine Info
# ============================================================================

def test_get_engine_info(aws_provider):
    """Test get_engine_info returns correct information."""
    name, version = aws_provider.get_engine_info()
    
    assert name == "cloud-aws-transcribe"
    assert version == "en-US-us-east-1"


def test_get_engine_info_with_different_config(aws_config):
    """Test get_engine_info with different configuration."""
    aws_config.language_code = "es-ES"
    aws_config.region = "eu-west-1"
    provider = CloudAWSTranscribeProvider(aws_config)
    
    name, version = provider.get_engine_info()
    
    assert name == "cloud-aws-transcribe"
    assert version == "es-ES-eu-west-1"


# ============================================================================
# Test: Get Model Info
# ============================================================================

def test_get_model_info(aws_provider):
    """Test get_model_info returns correct information."""
    info = aws_provider.get_model_info()
    
    assert info['language_code'] == 'en-US'
    assert info['region'] == 'us-east-1'
    assert info['max_file_size_gb'] == 2.0
    assert info['cost_per_minute_usd'] == 0.024
    assert info['credentials_configured'] is True
    assert info['implementation_status'] == 'fully_implemented'


def test_get_model_info_without_credentials(aws_config):
    """Test get_model_info when credentials are not configured."""
    aws_config.access_key_id = None
    aws_config.secret_access_key = None
    provider = CloudAWSTranscribeProvider(aws_config)
    
    info = provider.get_model_info()
    
    assert info['credentials_configured'] is False


# ============================================================================
# Test: Get File Size Limit
# ============================================================================

def test_get_file_size_limit(aws_provider):
    """Test get_file_size_limit returns correct limit."""
    limit = aws_provider.get_file_size_limit()
    
    assert limit == 2 * 1024 * 1024 * 1024  # 2 GB in bytes


# ============================================================================
# Test: Configuration Usage
# ============================================================================

@patch('time.sleep')
@patch('urllib.request.urlopen')
@patch('boto3.client')
def test_uses_config_language_code(mock_boto_client, mock_urlopen, mock_sleep, aws_config, mock_audio_file):
    """Test that provider uses language_code from config."""
    aws_config.language_code = "es-ES"
    provider = CloudAWSTranscribeProvider(aws_config)
    
    # Mock AWS clients
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    provider.client = mock_transcribe_client
    provider.s3_client = mock_s3_client
    
    # Mock S3 operations
    mock_s3_client.head_bucket.return_value = {}
    
    # Mock transcription job
    mock_transcribe_client.start_transcription_job.return_value = {}
    mock_transcribe_client.get_transcription_job.return_value = {
        'TranscriptionJob': {
            'TranscriptionJobStatus': 'COMPLETED',
            'Transcript': {
                'TranscriptFileUri': 'https://s3.amazonaws.com/bucket/transcript.json'
            }
        }
    }
    
    # Mock transcript download
    mock_response = Mock()
    mock_response.read.return_value = b'{"results": {}}'
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    mock_urlopen.return_value = mock_response
    
    provider.transcribe(mock_audio_file)
    
    # Verify correct language code was used
    call_kwargs = mock_transcribe_client.start_transcription_job.call_args[1]
    assert call_kwargs['LanguageCode'] == "es-ES"


@patch('boto3.client')
def test_uses_config_region(mock_boto_client, aws_config):
    """Test that provider uses region from config."""
    aws_config.region = "eu-west-1"
    provider = CloudAWSTranscribeProvider(aws_config)
    
    mock_transcribe_client = Mock()
    mock_s3_client = Mock()
    mock_boto_client.side_effect = [mock_transcribe_client, mock_s3_client]
    
    provider._ensure_client_initialized()
    
    # Verify correct region was used
    transcribe_call = mock_boto_client.call_args_list[0]
    assert transcribe_call[1]['region_name'] == "eu-west-1"


# ============================================================================
# Test: Pricing Configuration Override (Task 30.1)
# ============================================================================

def test_cost_per_minute_override_via_config():
    """Test that cost_per_minute_usd can be overridden via config."""
    config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket",
        cost_per_minute_usd=0.030  # Override default $0.024
    )
    provider = CloudAWSTranscribeProvider(config)
    
    # Verify config stores the override
    assert provider.config.cost_per_minute_usd == 0.030


def test_estimate_cost_uses_config_override():
    """Test that estimate_cost uses cost_per_minute_usd from config."""
    config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket",
        cost_per_minute_usd=0.030  # Override default $0.024
    )
    provider = CloudAWSTranscribeProvider(config)
    
    # 5 minutes = 300 seconds
    cost = provider.estimate_cost(300.0)
    
    # Cost should be 5 * $0.030 = $0.15 (not default $0.12)
    assert cost == 0.15


def test_get_model_info_returns_config_cost():
    """Test that get_model_info returns cost_per_minute_usd from config."""
    config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket",
        cost_per_minute_usd=0.030  # Override default $0.024
    )
    provider = CloudAWSTranscribeProvider(config)
    
    info = provider.get_model_info()
    
    assert info['cost_per_minute_usd'] == 0.030


def test_default_cost_per_minute_when_not_overridden():
    """Test that default cost_per_minute_usd is used when not overridden."""
    config = AWSTranscribeConfig(
        access_key_id="AKIATEST",
        secret_access_key="test_secret",
        region="us-east-1",
        s3_bucket="test-bucket"
        # cost_per_minute_usd not specified, should use default
    )
    provider = CloudAWSTranscribeProvider(config)
    
    # Verify default value is used
    assert provider.config.cost_per_minute_usd == 0.024
    
    # Verify estimate_cost uses default
    cost = provider.estimate_cost(60.0)  # 1 minute
    assert cost == 0.024


@patch.dict(os.environ, {'AWS_TRANSCRIBE_COST_PER_MINUTE': '0.035'})
def test_cost_per_minute_override_via_environment(tmp_path):
    """Test that cost_per_minute_usd can be overridden via environment variable."""
    from pipeline.transcription.config import TranscriptionConfig
    
    # Create a minimal config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
aws_transcribe:
  access_key_id: AKIATEST
  secret_access_key: test_secret
  region: us-east-1
  s3_bucket: test-bucket
""")
    
    # Load config with environment variable override
    full_config = TranscriptionConfig.load_from_yaml(str(config_file))
    config = full_config.aws_transcribe
    
    # Verify environment variable was applied
    assert config.cost_per_minute_usd == 0.035
    
    # Verify provider uses the override
    provider = CloudAWSTranscribeProvider(config)
    cost = provider.estimate_cost(60.0)  # 1 minute
    assert cost == 0.035
