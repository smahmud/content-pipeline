"""
File: base.py

Defines the enhanced TranscriberAdapter protocol for transcription adapters.
Used to enforce a consistent interface across adapter implementations.

Enhanced in v0.6.5 to support:
- Requirement validation before instantiation
- Supported format discovery
- Cost estimation for cloud services
- Standardized error handling
"""
from typing import Protocol, Optional, List, Tuple

class TranscriberAdapter(Protocol):
    """
    Enhanced protocol for transcription adapters.

    Implementations must provide:
    - Audio transcription from file path
    - Engine metadata reporting
    - Requirement validation
    - Supported format listing
    - Cost estimation (for paid services)
    """
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe the given audio file and return a raw transcript dictionary.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language hint for transcription
            
        Returns:
            Dictionary containing transcript data with 'text' key at minimum
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
            RuntimeError: If transcription fails
        """

    def get_engine_info(self) -> Tuple[str, str]:
        """
        Return the engine name and version used for transcription.
        
        Returns:
            Tuple of (engine_name, version_info)
        """

    def validate_requirements(self) -> List[str]:
        """
        Validate adapter requirements and return any errors.
        
        This method should check:
        - Required dependencies are installed
        - API keys/credentials are available and valid
        - Models are downloaded and accessible
        - Network connectivity (for cloud services)
        
        Returns:
            List of error messages. Empty list means all requirements are met.
        """

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats.
        
        Returns:
            List of supported file extensions (e.g., ['mp3', 'wav', 'flac'])
        """

    def estimate_cost(self, audio_duration: float) -> Optional[float]:
        """
        Estimate transcription cost in USD for the given audio duration.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated cost in USD, or None for free engines
        """
