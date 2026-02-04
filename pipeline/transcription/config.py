"""
Transcription Configuration Management

This module provides configuration classes for transcription providers.
It handles configuration loading from YAML files with environment variable
substitution and precedence rules.

Configuration Precedence (highest to lowest):
1. Explicit parameters passed to provider constructors
2. Environment variables (CONTENT_PIPELINE_*, OPENAI_API_KEY, AWS_*)
3. Project config (./.content-pipeline/config.yaml)
4. User config (~/.content-pipeline/config.yaml)
5. System defaults

**Validates: Requirements 2.5, 7.2, 7.4**
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class WhisperLocalConfig:
    """Configuration for local Whisper transcription provider.
    
    Attributes:
        model: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda, auto)
        compute_type: Compute type for faster-whisper
        timeout: Operation timeout in seconds
        retry_attempts: Number of retry attempts on failure
        retry_delay: Delay between retry attempts in seconds
    """
    model: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class WhisperAPIConfig:
    """Configuration for OpenAI Whisper API transcription provider.
    
    Attributes:
        api_key: OpenAI API key
        model: Whisper model to use (whisper-1)
        temperature: Sampling temperature (0.0 to 1.0)
        response_format: Response format (json, text, srt, verbose_json, vtt)
        timeout: API request timeout in seconds
        retry_attempts: Number of retry attempts on API failure
        retry_delay: Delay between retry attempts in seconds
        cost_per_minute_usd: Cost per minute in USD (default: 0.006, can be overridden for custom pricing)
    """
    api_key: Optional[str] = None
    model: str = "whisper-1"
    temperature: float = 0.0
    response_format: str = "json"
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    cost_per_minute_usd: float = 0.006


@dataclass
class AWSTranscribeConfig:
    """Configuration for AWS Transcribe transcription provider.
    
    Attributes:
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
        region: AWS region
        language_code: Language code for transcription (e.g., 'en-US')
        s3_bucket: Optional custom S3 bucket name
        timeout: Operation timeout in seconds
        retry_attempts: Number of retry attempts on failure
        retry_delay: Delay between retry attempts in seconds
        cost_per_minute_usd: Cost per minute in USD (default: 0.024, can be overridden for custom pricing)
    """
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: str = "us-east-1"
    language_code: str = "en-US"
    s3_bucket: Optional[str] = None
    timeout: int = 600
    retry_attempts: int = 3
    retry_delay: float = 2.0
    cost_per_minute_usd: float = 0.024


@dataclass
class TranscriptionConfig:
    """Main configuration class for transcription providers.
    
    This class aggregates all provider-specific configurations and provides
    methods for loading configuration from YAML files with environment
    variable substitution.
    
    Attributes:
        whisper_local: Configuration for local Whisper provider
        whisper_api: Configuration for OpenAI Whisper API provider
        aws_transcribe: Configuration for AWS Transcribe provider
    """
    whisper_local: WhisperLocalConfig = field(default_factory=WhisperLocalConfig)
    whisper_api: WhisperAPIConfig = field(default_factory=WhisperAPIConfig)
    aws_transcribe: AWSTranscribeConfig = field(default_factory=AWSTranscribeConfig)
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> "TranscriptionConfig":
        """Load configuration from YAML file with environment variable substitution.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            TranscriptionConfig instance with loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        
        # Extract provider-specific configurations
        whisper_local_data = config_data.get('whisper_local') or {}
        whisper_api_data = config_data.get('whisper_api') or {}
        aws_transcribe_data = config_data.get('aws_transcribe') or {}
        
        # Create provider configs with environment variable substitution
        whisper_local = WhisperLocalConfig(
            model=cls._resolve_value(whisper_local_data.get('model'), 'WHISPER_LOCAL_MODEL', 'base'),
            device=cls._resolve_value(whisper_local_data.get('device'), 'WHISPER_LOCAL_DEVICE', 'auto'),
            compute_type=cls._resolve_value(whisper_local_data.get('compute_type'), 'WHISPER_LOCAL_COMPUTE_TYPE', 'default'),
            timeout=int(cls._resolve_value(whisper_local_data.get('timeout'), 'WHISPER_LOCAL_TIMEOUT', '300')),
            retry_attempts=int(cls._resolve_value(whisper_local_data.get('retry_attempts'), 'WHISPER_LOCAL_RETRY_ATTEMPTS', '3')),
            retry_delay=float(cls._resolve_value(whisper_local_data.get('retry_delay'), 'WHISPER_LOCAL_RETRY_DELAY', '1.0')),
        )
        
        whisper_api = WhisperAPIConfig(
            api_key=cls._resolve_value(whisper_api_data.get('api_key'), 'OPENAI_API_KEY', None),
            model=cls._resolve_value(whisper_api_data.get('model'), 'WHISPER_API_MODEL', 'whisper-1'),
            temperature=float(cls._resolve_value(whisper_api_data.get('temperature'), 'WHISPER_API_TEMPERATURE', '0.0')),
            response_format=cls._resolve_value(whisper_api_data.get('response_format'), 'WHISPER_API_RESPONSE_FORMAT', 'json'),
            timeout=int(cls._resolve_value(whisper_api_data.get('timeout'), 'WHISPER_API_TIMEOUT', '60')),
            retry_attempts=int(cls._resolve_value(whisper_api_data.get('retry_attempts'), 'WHISPER_API_RETRY_ATTEMPTS', '3')),
            retry_delay=float(cls._resolve_value(whisper_api_data.get('retry_delay'), 'WHISPER_API_RETRY_DELAY', '2.0')),
            cost_per_minute_usd=float(cls._resolve_value(whisper_api_data.get('cost_per_minute_usd'), 'WHISPER_API_COST_PER_MINUTE', '0.006')),
        )
        
        aws_transcribe = AWSTranscribeConfig(
            access_key_id=cls._resolve_value(aws_transcribe_data.get('access_key_id'), 'AWS_ACCESS_KEY_ID', None),
            secret_access_key=cls._resolve_value(aws_transcribe_data.get('secret_access_key'), 'AWS_SECRET_ACCESS_KEY', None),
            region=cls._resolve_value(aws_transcribe_data.get('region'), 'AWS_DEFAULT_REGION', 'us-east-1'),
            language_code=cls._resolve_value(aws_transcribe_data.get('language_code'), 'AWS_TRANSCRIBE_LANGUAGE_CODE', 'en-US'),
            s3_bucket=cls._resolve_value(aws_transcribe_data.get('s3_bucket'), 'AWS_TRANSCRIBE_S3_BUCKET', None),
            timeout=int(cls._resolve_value(aws_transcribe_data.get('timeout'), 'AWS_TRANSCRIBE_TIMEOUT', '600')),
            retry_attempts=int(cls._resolve_value(aws_transcribe_data.get('retry_attempts'), 'AWS_TRANSCRIBE_RETRY_ATTEMPTS', '3')),
            retry_delay=float(cls._resolve_value(aws_transcribe_data.get('retry_delay'), 'AWS_TRANSCRIBE_RETRY_DELAY', '2.0')),
            cost_per_minute_usd=float(cls._resolve_value(aws_transcribe_data.get('cost_per_minute_usd'), 'AWS_TRANSCRIBE_COST_PER_MINUTE', '0.024')),
        )
        
        return cls(
            whisper_local=whisper_local,
            whisper_api=whisper_api,
            aws_transcribe=aws_transcribe
        )
    
    @staticmethod
    def _resolve_value(config_value: Any, env_var: str, default: Any) -> Any:
        """Resolve configuration value with precedence rules.
        
        Precedence (highest to lowest):
        1. Explicit config value (if not None and not empty string)
        2. Environment variable
        3. Default value
        
        Supports environment variable substitution syntax: ${VAR_NAME:-default}
        
        Args:
            config_value: Value from configuration file
            env_var: Environment variable name to check
            default: Default value if neither config nor env var is set
            
        Returns:
            Resolved configuration value
        """
        # Handle environment variable substitution syntax in config value
        if isinstance(config_value, str) and '${' in config_value:
            # Pattern: ${VAR_NAME:-default_value} or ${VAR_NAME}
            pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                var_default = match.group(2) if match.group(2) is not None else ''
                return os.getenv(var_name, var_default)
            
            config_value = re.sub(pattern, replace_env_var, config_value)
            
            # If after substitution the value is empty, treat as None
            if config_value == '':
                config_value = None
        
        # Apply precedence rules
        if config_value is not None and config_value != '':
            return config_value
        
        env_value = os.getenv(env_var)
        if env_value is not None and env_value != '':
            return env_value
        
        return default
