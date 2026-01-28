"""
File: factory.py

Engine factory pattern for creating transcription engine adapters.

Enhanced in v0.6.5 to support:
- Adapter registration and instantiation
- Requirement validation before instantiation
- Configuration passing to adapters
- Clear error messages for unsupported engines
"""
from typing import Dict, Type, List, Any, Optional
from pipeline.config.schema import TranscriptionConfig
from pipeline.transcribers.adapters.base import TranscriberAdapter


class EngineFactory:
    """
    Factory for creating transcription engine adapters.
    
    Supports registration of new engine types and validates requirements
    before instantiation to provide clear error messages.
    """
    
    def __init__(self):
        """Initialize the factory with default adapters."""
        self._adapters: Dict[str, Type[TranscriberAdapter]] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self) -> None:
        """Register the default engine adapters."""
        # Import here to avoid circular imports
        from pipeline.transcribers.adapters.local_whisper import LocalWhisperAdapter
        from pipeline.transcribers.adapters.openai_whisper import OpenAIWhisperAdapter
        from pipeline.transcribers.adapters.aws_transcribe import AWSTranscribeAdapter
        
        # Register the local Whisper adapter
        self._adapters['local-whisper'] = LocalWhisperAdapter
        
        # Register the OpenAI Whisper API adapter
        self._adapters['openai-whisper'] = OpenAIWhisperAdapter
        
        # Register the AWS Transcribe adapter
        self._adapters['aws-transcribe'] = AWSTranscribeAdapter
    
    def create_engine(self, engine_type: str, config: TranscriptionConfig) -> TranscriberAdapter:
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
    
    def _create_adapter_instance(self, adapter_class: Type[TranscriberAdapter], 
                                engine_type: str, config: TranscriptionConfig) -> TranscriberAdapter:
        """
        Create an adapter instance with the appropriate configuration.
        
        Args:
            adapter_class: The adapter class to instantiate
            engine_type: Type of engine being created
            config: Complete transcription configuration
            
        Returns:
            Configured adapter instance
        """
        if engine_type == 'local-whisper':
            return adapter_class(
                model_name=config.whisper_local.model,
                device=config.whisper_local.device
            )
        elif engine_type == 'openai-whisper':
            # Create OpenAIWhisperAdapter with configuration
            api_key = config.whisper_api.api_key or self._get_api_key_from_env()
            return adapter_class(
                api_key=api_key,
                model=config.whisper_api.model,
                temperature=config.whisper_api.temperature,
                response_format=config.whisper_api.response_format
            )
        elif engine_type == 'aws-transcribe':
            # Create AWSTranscribeAdapter with configuration
            kwargs = {
                'access_key_id': config.aws_transcribe.access_key_id,
                'secret_access_key': config.aws_transcribe.secret_access_key,
                'region': config.aws_transcribe.region,
                'language_code': config.aws_transcribe.language_code
            }
            # Add s3_bucket if specified
            if config.aws_transcribe.s3_bucket:
                kwargs['s3_bucket'] = config.aws_transcribe.s3_bucket
            return adapter_class(**kwargs)
        else:
            # Generic instantiation for custom adapters
            return adapter_class()
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get OpenAI API key from environment variables."""
        import os
        return os.getenv('OPENAI_API_KEY')
    
    def register_adapter(self, engine_type: str, adapter_class: Type[TranscriberAdapter]) -> None:
        """
        Register a new engine adapter type.
        
        Args:
            engine_type: Unique identifier for the engine type
            adapter_class: Adapter class that implements TranscriberAdapter protocol
            
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
    
    def validate_engine_requirements(self, engine_type: str, config: TranscriptionConfig) -> List[str]:
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