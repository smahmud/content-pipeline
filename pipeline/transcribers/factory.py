"""
File: factory.py

Engine factory pattern for creating transcription engine adapters.

Enhanced in v0.6.5 to support:
- Adapter registration and instantiation
- Requirement validation before instantiation
- Configuration passing to adapters
- Clear error messages for unsupported engines

DEPRECATED: This module is deprecated. Use pipeline.transcription.TranscriptionProviderFactory instead.
"""
from typing import Dict, Type, List, Any, Optional
from pipeline.config.schema import TranscriptionConfig as OldTranscriptionConfig
from pipeline.transcription import (
    TranscriberProvider,
    WhisperLocalConfig,
    WhisperAPIConfig,
    AWSTranscribeConfig
)


class EngineFactory:
    """
    Factory for creating transcription engine adapters.
    
    Supports registration of new engine types and validates requirements
    before instantiation to provide clear error messages.
    
    DEPRECATED: Use pipeline.transcription.TranscriptionProviderFactory instead.
    """
    
    def __init__(self):
        """Initialize the factory with default adapters."""
        self._adapters: Dict[str, Type[TranscriberProvider]] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self) -> None:
        """Register the default engine adapters."""
        # Import here to avoid circular imports
        from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
        from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
        from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider
        
        # Register the local Whisper adapter
        self._adapters['local-whisper'] = LocalWhisperProvider
        
        # Register the OpenAI Whisper API adapter
        self._adapters['openai-whisper'] = CloudOpenAIWhisperProvider
        
        # Register the AWS Transcribe adapter
        self._adapters['aws-transcribe'] = CloudAWSTranscribeProvider
    
    def create_engine(self, engine_type: str, config: OldTranscriptionConfig) -> TranscriberProvider:
        """
        Create and configure the specified engine adapter.
        
        Args:
            engine_type: Type of engine to create (e.g., 'local-whisper', 'openai-whisper')
            config: Complete transcription configuration
            
        Returns:
            Configured transcription adapter instance
            
        Raises:
            ValueError: If engine type is not supported
            RuntimeError: If engine requirements are not met
        """
        if engine_type not in self._adapters:
            available_engines = list(self._adapters.keys())
            raise ValueError(f"Unsupported engine type '{engine_type}'. Available engines: {available_engines}")
        
        adapter_class = self._adapters[engine_type]
        
        # Create adapter instance with engine-specific configuration
        adapter = self._create_adapter_instance(adapter_class, engine_type, config)
        
        # Validate requirements before returning
        errors = adapter.validate_requirements()
        if errors:
            error_msg = f"Engine '{engine_type}' requirements not met:\n" + "\n".join(f"  - {error}" for error in errors)
            raise RuntimeError(error_msg)
        
        return adapter
    
    def _create_adapter_instance(self, adapter_class: Type[TranscriberProvider], 
                                engine_type: str, config: OldTranscriptionConfig) -> TranscriberProvider:
        """
        Create an adapter instance with the appropriate configuration.
        
        Converts old config format to new config format for providers.
        
        Args:
            adapter_class: The adapter class to instantiate
            engine_type: Type of engine being created
            config: Complete transcription configuration (old format)
            
        Returns:
            Configured adapter instance
        """
        if engine_type == 'local-whisper':
            # Convert old config to new config format
            new_config = WhisperLocalConfig(
                model=config.whisper_local.model,
                device=config.whisper_local.device,
                compute_type=getattr(config.whisper_local, 'compute_type', 'default'),
                timeout=getattr(config.whisper_local, 'timeout', 300),
                retry_attempts=getattr(config.whisper_local, 'retry_attempts', 3),
                retry_delay=getattr(config.whisper_local, 'retry_delay', 1.0)
            )
            return adapter_class(new_config)
        elif engine_type == 'openai-whisper':
            # Convert old config to new config format
            new_config = WhisperAPIConfig(
                api_key=config.whisper_api.api_key,
                model=config.whisper_api.model,
                temperature=config.whisper_api.temperature,
                response_format=config.whisper_api.response_format,
                timeout=getattr(config.whisper_api, 'timeout', 300),
                retry_attempts=getattr(config.whisper_api, 'retry_attempts', 3),
                retry_delay=getattr(config.whisper_api, 'retry_delay', 1.0)
            )
            return adapter_class(new_config)
        elif engine_type == 'aws-transcribe':
            # Convert old config to new config format
            new_config = AWSTranscribeConfig(
                access_key_id=config.aws_transcribe.access_key_id,
                secret_access_key=config.aws_transcribe.secret_access_key,
                region=config.aws_transcribe.region,
                language_code=config.aws_transcribe.language_code,
                s3_bucket=config.aws_transcribe.s3_bucket,
                timeout=getattr(config.aws_transcribe, 'timeout', 300),
                retry_attempts=getattr(config.aws_transcribe, 'retry_attempts', 3),
                retry_delay=getattr(config.aws_transcribe, 'retry_delay', 1.0)
            )
            return adapter_class(new_config)
        else:
            # Generic instantiation for custom adapters
            return adapter_class()
    
    def register_adapter(self, engine_type: str, adapter_class: Type[TranscriberProvider]) -> None:
        """
        Register a new engine adapter type.
        
        Args:
            engine_type: Unique identifier for the engine type
            adapter_class: Adapter class that implements TranscriberProvider protocol
            
        Raises:
            ValueError: If engine_type is already registered
        """
        if engine_type in self._adapters:
            raise ValueError(f"Engine type '{engine_type}' is already registered")
        
        self._adapters[engine_type] = adapter_class
    
    def get_available_engines(self) -> List[str]:
        """
        Return list of available engine types.
        
        Returns:
            List of registered engine type identifiers
        """
        return list(self._adapters.keys())
    
    def validate_engine_requirements(self, engine_type: str, config: OldTranscriptionConfig) -> List[str]:
        """
        Validate that engine requirements are met without creating the adapter.
        
        Args:
            engine_type: Type of engine to validate
            config: Complete transcription configuration
            
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type not in self._adapters:
            available_engines = list(self._adapters.keys())
            return [f"Unsupported engine type '{engine_type}'. Available engines: {available_engines}"]
        
        try:
            adapter_class = self._adapters[engine_type]
            adapter = self._create_adapter_instance(adapter_class, engine_type, config)
            return adapter.validate_requirements()
        except Exception as e:
            return [f"Failed to validate engine '{engine_type}': {e}"]
    
    def is_engine_available(self, engine_type: str) -> bool:
        """
        Check if an engine type is available (registered).
        
        Args:
            engine_type: Type of engine to check
            
        Returns:
            True if engine is registered, False otherwise
        """
        return engine_type in self._adapters
    
    def get_engine_info(self, engine_type: str) -> Dict[str, Any]:
        """
        Get information about a registered engine.
        
        Args:
            engine_type: Type of engine to get info for
            
        Returns:
            Dictionary with engine information
            
        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type not in self._adapters:
            available_engines = list(self._adapters.keys())
            raise ValueError(f"Unsupported engine type '{engine_type}'. Available engines: {available_engines}")
        
        adapter_class = self._adapters[engine_type]
        
        return {
            'engine_type': engine_type,
            'adapter_class': adapter_class.__name__,
            'module': adapter_class.__module__,
            'is_available': True
        }