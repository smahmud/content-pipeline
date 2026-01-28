"""
Configuration schema and data models for Enhanced Transcription & Configuration v0.6.5.

This module defines the configuration data structures for transcription engines:
- Local Whisper models
- OpenAI Whisper API  
- AWS Transcribe service
- Auto-selection logic
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class EngineType(Enum):
    """Supported transcription engine types."""
    WHISPER_LOCAL = "whisper-local"
    WHISPER_API = "whisper-api"
    AWS_TRANSCRIBE = "aws-transcribe"
    AUTO = "auto"


class LogLevel(Enum):
    """Supported logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WhisperModelSize(Enum):
    """Supported Whisper model sizes for local processing."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class EngineConfig:
    """Base configuration for transcription engines."""
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class WhisperLocalConfig(EngineConfig):
    """Configuration for local Whisper engine."""
    model: str = WhisperModelSize.BASE.value
    device: str = "auto"  # 'cpu', 'cuda', 'auto'
    compute_type: str = "default"


@dataclass
class WhisperAPIConfig(EngineConfig):
    """Configuration for OpenAI Whisper API."""
    api_key: Optional[str] = None
    model: str = "whisper-1"
    temperature: float = 0.0
    response_format: str = "json"


@dataclass
class AWSTranscribeConfig(EngineConfig):
    """Configuration for AWS Transcribe service."""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: str = "us-east-1"
    language_code: str = "en-US"
    s3_bucket: Optional[str] = None  # Optional custom S3 bucket name
    media_format: str = "auto"  # auto-detect or specify: mp3, mp4, wav, etc.


@dataclass
class AutoSelectionConfig:
    """Configuration for auto engine selection."""
    prefer_local: bool = True
    fallback_enabled: bool = True
    priority_order: List[str] = field(default_factory=lambda: [
        EngineType.WHISPER_LOCAL.value,
        EngineType.AWS_TRANSCRIBE.value,
        EngineType.WHISPER_API.value
    ])


@dataclass
class TranscriptionConfig:
    """Complete transcription configuration for v0.6.5."""
    
    # Core settings
    engine: str = EngineType.AUTO.value
    output_dir: str = "./transcripts"
    log_level: str = LogLevel.INFO.value
    language: Optional[str] = None
    
    # Engine-specific configurations
    whisper_local: WhisperLocalConfig = field(default_factory=WhisperLocalConfig)
    whisper_api: WhisperAPIConfig = field(default_factory=WhisperAPIConfig)
    aws_transcribe: AWSTranscribeConfig = field(default_factory=AWSTranscribeConfig)
    
    # Auto-selection preferences
    auto_selection: AutoSelectionConfig = field(default_factory=AutoSelectionConfig)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate engine type
        try:
            EngineType(self.engine)
        except ValueError:
            valid_engines = [e.value for e in EngineType]
            errors.append(f"Invalid engine '{self.engine}'. Valid options: {valid_engines}")
        
        # Validate log level
        try:
            LogLevel(self.log_level)
        except ValueError:
            valid_levels = [l.value for l in LogLevel]
            errors.append(f"Invalid log_level '{self.log_level}'. Valid options: {valid_levels}")
        
        # Validate Whisper model size
        if self.whisper_local.model not in [m.value for m in WhisperModelSize]:
            valid_models = [m.value for m in WhisperModelSize]
            errors.append(f"Invalid whisper_local.model '{self.whisper_local.model}'. Valid options: {valid_models}")
        
        # Validate timeout values
        if self.whisper_local.timeout <= 0:
            errors.append("whisper_local.timeout must be positive")
        if self.whisper_api.timeout <= 0:
            errors.append("whisper_api.timeout must be positive")
        if self.aws_transcribe.timeout <= 0:
            errors.append("aws_transcribe.timeout must be positive")
        
        # Validate retry attempts
        if self.whisper_local.retry_attempts < 0:
            errors.append("whisper_local.retry_attempts must be non-negative")
        if self.whisper_api.retry_attempts < 0:
            errors.append("whisper_api.retry_attempts must be non-negative")
        if self.aws_transcribe.retry_attempts < 0:
            errors.append("aws_transcribe.retry_attempts must be non-negative")
        
        return errors
    
    def get_engine_config(self, engine_type: str) -> EngineConfig:
        """Get configuration for specific engine type."""
        if engine_type == EngineType.WHISPER_LOCAL.value:
            return self.whisper_local
        elif engine_type == EngineType.WHISPER_API.value:
            return self.whisper_api
        elif engine_type == EngineType.AWS_TRANSCRIBE.value:
            return self.aws_transcribe
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")