"""
Unit tests for AWSTranscribeAdapter behavior and error handling.

Covers:
- AWS Transcribe service integration (placeholder implementation)
- AWS credentials validation and authentication
- Cost estimation and file size limits
- Enhanced error handling and validation
- Protocol compliance and configuration
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipeline.transcribers.adapters.aws_transcribe import AWSTranscribeAdapter
from pipeline.transcribers.adapters.base import TranscriberAdapter


class TestAWSTranscribeAdapter:
    """Test the AWSTranscribeAdapter class."""

    def test_adapter_implements_enhanced_protocol(self):
        """Test that AWSTranscribeAdapter implements all enhanced protocol methods."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Check that all protocol methods exist
        assert hasattr(adapter, 'transcribe')
        assert hasattr(adapter, 'get_engine_info')
        assert hasattr(adapter, 'validate_requirements')
        assert hasattr(adapter, 'get_supported_formats')
        assert hasattr(adapter, 'estimate_cost')
        
        # Check that methods are callable
        assert callable(adapter.transcribe)
        assert callable(adapter.get_engine_info)
        assert callable(adapter.validate_requirements)
        assert callable(adapter.get_supported_formats)
        assert callable(adapter.estimate_cost)

    def test_adapter_complies_with_protocol(self):
        """Test that AWSTranscribeAdapter is recognized as a TranscriberAdapter."""
        adapter: TranscriberAdapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        engine, version = adapter.get_engine_info()
        assert isinstance(engine, str)
        assert isinstance(version, str)
        assert engine == "aws-transcribe"

    def test_initialization_with_default_parameters(self):
        """Test adapter initialization with default parameters."""
        with patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'}):
            adapter = AWSTranscribeAdapter()
            
            assert adapter.access_key_id == "test_key"
            assert adapter.secret_access_key == "test_secret"
            assert adapter.region == "us-east-1"
            assert adapter.language_code == "en-US"
            assert adapter.client is None

    def test_initialization_with_custom_parameters(self):
        """Test adapter initialization with custom parameters."""
        adapter = AWSTranscribeAdapter(
            access_key_id="custom_key",
            secret_access_key="custom_secret",
            region="eu-west-1",
            language_code="es-ES"
        )
        
        assert adapter.access_key_id == "custom_key"
        assert adapter.secret_access_key == "custom_secret"
        assert adapter.region == "eu-west-1"
        assert adapter.language_code == "es-ES"
        assert adapter.client is None

    def test_get_credentials_from_environment(self):
        """Test credential retrieval from environment variables."""
        # Test AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        with patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'env_key', 'AWS_SECRET_ACCESS_KEY': 'env_secret'}):
            adapter = AWSTranscribeAdapter()
            assert adapter.access_key_id == "env_key"
            assert adapter.secret_access_key == "env_secret"
        
        # Test CONTENT_PIPELINE_* variables
        with patch.dict(os.environ, {
            'CONTENT_PIPELINE_AWS_ACCESS_KEY_ID': 'pipeline_key',
            'CONTENT_PIPELINE_AWS_SECRET_ACCESS_KEY': 'pipeline_secret'
        }, clear=True):
            adapter = AWSTranscribeAdapter()
            assert adapter.access_key_id == "pipeline_key"
            assert adapter.secret_access_key == "pipeline_secret"
        
        # Test no environment variables
        with patch.dict(os.environ, {}, clear=True):
            adapter = AWSTranscribeAdapter()
            assert adapter.access_key_id is None
            assert adapter.secret_access_key is None

    def test_get_supported_formats_returns_list(self):
        """Test that get_supported_formats returns a list of strings."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        formats = adapter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        
        # Check for expected AWS Transcribe formats
        expected_formats = ['mp3', 'wav', 'flac', 'mp4']
        for fmt in expected_formats:
            assert fmt in formats

    def test_estimate_cost_calculation(self):
        """Test cost estimation calculation."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Test 1 minute of audio
        cost = adapter.estimate_cost(60.0)
        assert cost == 0.024  # $0.024 per minute
        
        # Test 30 seconds of audio
        cost = adapter.estimate_cost(30.0)
        assert cost == 0.012  # Half a minute
        
        # Test 2.5 minutes of audio
        cost = adapter.estimate_cost(150.0)
        assert cost == 0.06  # 2.5 * $0.024
        
        # Test zero duration
        cost = adapter.estimate_cost(0.0)
        assert cost == 0.0
        
        # Test negative duration
        cost = adapter.estimate_cost(-10.0)
        assert cost == 0.0

    def test_get_engine_info_returns_correct_info(self):
        """Test that get_engine_info returns correct engine information."""
        adapter = AWSTranscribeAdapter(
            access_key_id="test_key", 
            secret_access_key="test_secret",
            region="eu-west-1",
            language_code="fr-FR"
        )
        engine_info = adapter.get_engine_info()
        
        assert isinstance(engine_info, tuple)
        assert len(engine_info) == 2
        assert engine_info[0] == "aws-transcribe"
        assert engine_info[1] == "fr-FR-eu-west-1"

    def test_get_model_info(self):
        """Test that get_model_info returns model information."""
        adapter = AWSTranscribeAdapter(
            access_key_id="test_key",
            secret_access_key="test_secret",
            region="us-west-2",
            language_code="en-US"
        )
        info = adapter.get_model_info()
        
        assert isinstance(info, dict)
        assert info['language_code'] == 'en-US'
        assert info['region'] == 'us-west-2'
        assert info['max_file_size_gb'] == 2
        assert info['cost_per_minute_usd'] == 0.024
        assert info['credentials_configured'] is True
        assert info['implementation_status'] == 'placeholder'

    def test_get_file_size_limit(self):
        """Test file size limit retrieval."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        limit = adapter.get_file_size_limit()
        
        assert limit == 2 * 1024 * 1024 * 1024  # 2 GB in bytes

    def test_validate_requirements_missing_boto3_package(self):
        """Test validate_requirements when boto3 package is not installed."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Mock import error for boto3 package
        with patch('builtins.__import__', side_effect=ImportError("No module named 'boto3'")):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("boto3 package not installed" in error for error in errors)

    def test_validate_requirements_missing_credentials(self):
        """Test validate_requirements when credentials are missing."""
        # Test missing access key
        adapter = AWSTranscribeAdapter(access_key_id=None, secret_access_key="test_secret")
        
        # Mock boto3 to be available
        with patch('pipeline.transcribers.adapters.aws_transcribe.boto3'):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("AWS access key not found" in error for error in errors)
        
        # Test missing secret key
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key=None)
        
        with patch('pipeline.transcribers.adapters.aws_transcribe.boto3'):
            errors = adapter.validate_requirements()
            
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("AWS secret key not found" in error for error in errors)

    @patch('pipeline.transcribers.adapters.aws_transcribe.boto3')
    def test_validate_requirements_success_with_implementation_warning(self, mock_boto3):
        """Test validate_requirements when all requirements are met but implementation is placeholder."""
        # Mock successful boto3 client creation
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) == 1  # Only the implementation warning
        assert any("not yet fully implemented" in error for error in errors)

    @patch('pipeline.transcribers.adapters.aws_transcribe.boto3')
    def test_validate_requirements_client_initialization_failure(self, mock_boto3):
        """Test validate_requirements when client initialization fails."""
        mock_boto3.client.side_effect = Exception("AWS connection failed")
        
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        errors = adapter.validate_requirements()
        
        assert isinstance(errors, list)
        assert len(errors) >= 1
        assert any("Failed to initialize AWS Transcribe client" in error for error in errors)

    def test_transcribe_validates_file_existence(self):
        """Test that transcribe method validates file existence."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            adapter.transcribe("nonexistent_file.mp3")

    def test_transcribe_validates_file_format(self):
        """Test that transcribe method validates file format."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Create a temporary file with unsupported extension
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported audio format"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_transcribe_validates_file_size(self):
        """Test that transcribe method validates file size."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Mock os.path.getsize to return a large file size
        with patch('os.path.getsize', return_value=3 * 1024 * 1024 * 1024):  # 3 GB
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                with pytest.raises(ValueError, match="File too large"):
                    adapter.transcribe(temp_file_path)
            finally:
                os.unlink(temp_file_path)

    @patch('pipeline.transcribers.adapters.aws_transcribe.boto3')
    def test_transcribe_not_implemented_error(self, mock_boto3):
        """Test that transcribe raises NotImplementedError for placeholder implementation."""
        # Mock successful boto3 client creation
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Create a temporary valid audio file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(NotImplementedError, match="AWS Transcribe adapter is not yet fully implemented"):
                adapter.transcribe(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_supported_formats_immutability(self):
        """Test that supported formats list cannot be modified externally."""
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        formats1 = adapter.get_supported_formats()
        formats2 = adapter.get_supported_formats()
        
        # Should return different list instances
        assert formats1 is not formats2
        
        # Modifying one shouldn't affect the other
        formats1.append("fake_format")
        assert "fake_format" not in formats2
        assert "fake_format" not in adapter.get_supported_formats()

    def test_configuration_storage(self):
        """Test that additional configuration is stored correctly."""
        config_kwargs = {
            'timeout': 300,
            'retry_attempts': 3,
            'custom_param': 'test_value'
        }
        
        adapter = AWSTranscribeAdapter(
            access_key_id="test_key", 
            secret_access_key="test_secret",
            **config_kwargs
        )
        
        assert adapter.config == config_kwargs
        assert adapter.config['timeout'] == 300
        assert adapter.config['retry_attempts'] == 3
        assert adapter.config['custom_param'] == 'test_value'


class TestAWSTranscribeAdapterIntegration:
    """Integration tests for AWSTranscribeAdapter."""

    @patch('pipeline.transcribers.adapters.aws_transcribe.boto3')
    def test_full_adapter_lifecycle(self, mock_boto3):
        """Test the complete lifecycle of adapter usage."""
        # Mock successful setup
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        adapter = AWSTranscribeAdapter(access_key_id="test_key", secret_access_key="test_secret")
        
        # Validate requirements (should have implementation warning)
        errors = adapter.validate_requirements()
        assert len(errors) == 1  # Only implementation warning
        assert "not yet fully implemented" in errors[0]
        
        # Get engine info
        engine, version = adapter.get_engine_info()
        assert engine == "aws-transcribe"
        assert "en-US-us-east-1" in version
        
        # Get supported formats
        formats = adapter.get_supported_formats()
        assert 'mp3' in formats
        assert 'wav' in formats
        
        # Estimate cost
        cost = adapter.estimate_cost(60.0)
        assert cost == 0.024

    def test_different_regions_and_languages(self):
        """Test adapter with different region and language configurations."""
        configs = [
            ('us-east-1', 'en-US'),
            ('eu-west-1', 'en-GB'),
            ('ap-southeast-1', 'zh-CN'),
            ('us-west-2', 'es-ES')
        ]
        
        for region, language_code in configs:
            adapter = AWSTranscribeAdapter(
                access_key_id="test_key",
                secret_access_key="test_secret",
                region=region,
                language_code=language_code
            )
            
            # Check that configuration is set correctly
            assert adapter.region == region
            assert adapter.language_code == language_code
            
            # Check engine info
            engine, version = adapter.get_engine_info()
            assert engine == "aws-transcribe"
            assert f"{language_code}-{region}" in version
            
            # Check model info
            info = adapter.get_model_info()
            assert info['region'] == region
            assert info['language_code'] == language_code