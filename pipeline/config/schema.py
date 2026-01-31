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
    WHISPER_LOCAL = "local-whisper"
    WHISPER_API = "openai-whisper"
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


class LLMProvider(Enum):
    """Supported LLM providers for enrichment."""
    OPENAI = "openai"
    CLAUDE = "claude"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"
    AUTO = "auto"


class QualityPreset(Enum):
    """Quality presets for enrichment."""
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


class ContentProfile(Enum):
    """Content profiles for enrichment."""
    PODCAST = "podcast"
    MEETING = "meeting"
    LECTURE = "lecture"
    CUSTOM = "custom"


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



# ============================================================================
# Enrichment Configuration (v0.7.0)
# ============================================================================

@dataclass
class EnrichmentProviderConfig:
    """Configuration for a specific LLM provider.
    
    Attributes:
        enabled: Whether this provider is enabled
        api_key: API key for the provider (if applicable)
        default_model: Default model to use for this provider
        max_cost_per_request: Maximum cost per request in USD
    """
    enabled: bool = True
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    max_cost_per_request: Optional[float] = None


@dataclass
class EnrichmentCacheConfig:
    """Configuration for enrichment caching.
    
    Attributes:
        enabled: Whether caching is enabled
        ttl_days: Time-to-live for cache entries in days
        max_size_mb: Maximum cache size in megabytes
        cache_dir: Directory for cache files
    """
    enabled: bool = True
    ttl_days: int = 30
    max_size_mb: int = 500
    cache_dir: str = ".content-pipeline/cache/enrichment"


@dataclass
class EnrichmentCostControlConfig:
    """Configuration for cost control.
    
    Attributes:
        max_cost_per_request: Maximum cost per enrichment request in USD
        max_cost_per_batch: Maximum cost per batch operation in USD
        warning_threshold: Threshold for cost warnings (0.5 = 50%)
        require_confirmation: Whether to require confirmation before proceeding
    """
    max_cost_per_request: Optional[float] = None
    max_cost_per_batch: Optional[float] = None
    warning_threshold: float = 0.5
    require_confirmation: bool = False


@dataclass
class EnrichmentConfig:
    """Complete enrichment configuration for v0.7.0.
    
    This configuration defines settings for LLM-powered enrichment including
    provider configurations, cost control, caching, and default preferences.
    
    Attributes:
        provider: Default LLM provider to use
        quality: Default quality preset
        content_profile: Default content profile
        enrichment_types: Default enrichment types to enable
        custom_prompts_dir: Optional custom prompts directory
        providers: Provider-specific configurations
        cache: Cache configuration
        cost_control: Cost control configuration
    """
    # Default settings
    provider: str = LLMProvider.AUTO.value
    quality: str = QualityPreset.BALANCED.value
    content_profile: Optional[str] = None
    enrichment_types: List[str] = field(default_factory=lambda: ["summary", "tag"])
    custom_prompts_dir: Optional[str] = None
    
    # Provider configurations
    openai: EnrichmentProviderConfig = field(default_factory=EnrichmentProviderConfig)
    claude: EnrichmentProviderConfig = field(default_factory=EnrichmentProviderConfig)
    bedrock: EnrichmentProviderConfig = field(default_factory=EnrichmentProviderConfig)
    ollama: EnrichmentProviderConfig = field(default_factory=EnrichmentProviderConfig)
    
    # Cache configuration
    cache: EnrichmentCacheConfig = field(default_factory=EnrichmentCacheConfig)
    
    # Cost control configuration
    cost_control: EnrichmentCostControlConfig = field(default_factory=EnrichmentCostControlConfig)
    
    def validate(self) -> List[str]:
        """Validate enrichment configuration and return list of errors."""
        errors = []
        
        # Validate provider
        try:
            LLMProvider(self.provider)
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            errors.append(f"Invalid provider '{self.provider}'. Valid options: {valid_providers}")
        
        # Validate quality preset
        try:
            QualityPreset(self.quality)
        except ValueError:
            valid_presets = [q.value for q in QualityPreset]
            errors.append(f"Invalid quality '{self.quality}'. Valid options: {valid_presets}")
        
        # Validate content profile if specified
        if self.content_profile:
            try:
                ContentProfile(self.content_profile)
            except ValueError:
                valid_profiles = [p.value for p in ContentProfile]
                errors.append(f"Invalid content_profile '{self.content_profile}'. Valid options: {valid_profiles}")
        
        # Validate enrichment types
        valid_types = ["summary", "tag", "chapter", "highlight"]
        for etype in self.enrichment_types:
            if etype not in valid_types:
                errors.append(f"Invalid enrichment type '{etype}'. Valid options: {valid_types}")
        
        # Validate cache settings
        if self.cache.ttl_days <= 0:
            errors.append("cache.ttl_days must be positive")
        if self.cache.max_size_mb <= 0:
            errors.append("cache.max_size_mb must be positive")
        
        # Validate cost control settings
        if self.cost_control.max_cost_per_request is not None and self.cost_control.max_cost_per_request <= 0:
            errors.append("cost_control.max_cost_per_request must be positive")
        if self.cost_control.max_cost_per_batch is not None and self.cost_control.max_cost_per_batch <= 0:
            errors.append("cost_control.max_cost_per_batch must be positive")
        if not (0.0 <= self.cost_control.warning_threshold <= 1.0):
            errors.append("cost_control.warning_threshold must be between 0.0 and 1.0")
        
        return errors
    
    def get_provider_config(self, provider: str) -> EnrichmentProviderConfig:
        """Get configuration for specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration
            
        Raises:
            ValueError: If provider is unknown
        """
        if provider == LLMProvider.OPENAI.value:
            return self.openai
        elif provider == LLMProvider.CLAUDE.value:
            return self.claude
        elif provider == LLMProvider.BEDROCK.value:
            return self.bedrock
        elif provider == LLMProvider.OLLAMA.value:
            return self.ollama
        else:
            raise ValueError(f"Unknown provider: {provider}")


@dataclass
class PipelineConfig:
    """Complete pipeline configuration combining transcription and enrichment.
    
    This is the top-level configuration that includes both transcription
    and enrichment settings.
    
    Attributes:
        transcription: Transcription configuration
        enrichment: Enrichment configuration
    """
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    
    def validate(self) -> List[str]:
        """Validate complete pipeline configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        errors.extend(self.transcription.validate())
        errors.extend(self.enrichment.validate())
        return errors
