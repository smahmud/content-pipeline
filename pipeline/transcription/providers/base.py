"""
Transcription Provider Protocol

Defines the TranscriberProvider protocol for transcription provider implementations.
Used to enforce a consistent interface across provider implementations.

This module was migrated from pipeline.transcribers.adapters.base as part of the
infrastructure refactoring to establish a clean separation between infrastructure
and domain layers.

Enhanced to support:
- Requirement validation before instantiation
- Supported format discovery
- Cost estimation for cloud services
- Standardized error handling

**Validates: Requirements 2.2, 2.9**
"""
from typing import Protocol, Optional, List, Tuple, runtime_checkable


@runtime_checkable
class TranscriberProvider(Protocol):
    """
    Protocol for transcription provider implementations.

    All transcription providers must implement this protocol to ensure
    consistent behavior across local and cloud-based transcription services.

    Implementations must provide:
    - Audio transcription from file path
    - Provider metadata reporting
    - Requirement validation
    - Supported format listing
    - Cost estimation (for paid services)
    
    Example:
        >>> provider = LocalWhisperProvider(config)
        >>> errors = provider.validate_requirements()
        >>> if not errors:
        ...     result = provider.transcribe("audio.mp3", language="en")
        ...     print(result['text'])
    """
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe the given audio file and return a raw transcript dictionary.
        
        This method processes an audio file and returns the transcribed text
        along with any additional metadata provided by the transcription service.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription (e.g., 'en', 'es')
            
        Returns:
            Dictionary containing transcript data with 'text' key at minimum.
            May include additional keys like 'segments', 'language', 'confidence'.
            
        Raises:
            AudioFileError: If audio file doesn't exist or is invalid
            ProviderError: If transcription fails
            TranscriptionTimeoutError: If transcription exceeds timeout
            
        Example:
            >>> result = provider.transcribe("audio.mp3", language="en")
            >>> print(result['text'])
            "This is the transcribed text..."
        """
        ...

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the provider name and version used for transcription.
        
        This information is used for metadata tracking and debugging purposes.
        
        Returns:
            Tuple of (provider_name, version_info)
            
        Example:
            >>> name, version = provider.get_engine_info()
            >>> print(f"{name} v{version}")
            "local-whisper v3.0"
        """
        ...

    def validate_requirements(self) -> List[str]:
        """
        Validate provider requirements and return any errors.
        
        This method should check all prerequisites before attempting transcription:
        - Required dependencies are installed
        - API keys/credentials are available and valid
        - Models are downloaded and accessible
        - Network connectivity (for cloud services)
        - Sufficient disk space (for local models)
        
        Returns:
            List of error messages. Empty list means all requirements are met.
            
        Example:
            >>> errors = provider.validate_requirements()
            >>> if errors:
            ...     print("Provider not ready:", errors)
            ... else:
            ...     print("Provider ready to use")
        """
        ...

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats.
        
        This method returns the file extensions that this provider can process.
        Used for validation before attempting transcription.
        
        Returns:
            List of supported file extensions (e.g., ['mp3', 'wav', 'flac'])
            
        Example:
            >>> formats = provider.get_supported_formats()
            >>> if 'mp3' in formats:
            ...     print("MP3 files are supported")
        """
        ...

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost in USD for the given audio duration.
        
        This method calculates the expected cost before transcription,
        allowing users to make informed decisions about whether to proceed.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD, or None for free providers (local models)
            
        Example:
            >>> cost = provider.estimate_cost(300.0)  # 5 minutes
            >>> if cost:
            ...     print(f"Estimated cost: ${cost:.4f}")
            ... else:
            ...     print("Free transcription")
        """
        ...
