"""
Transcription Error Classes

This module defines custom exception classes for transcription operations.
These exceptions provide clear error messages and enable proper error handling
throughout the transcription infrastructure.

**Validates: Requirements 2.6**
"""


class TranscriptionError(Exception):
    """Base exception class for all transcription-related errors.
    
    This is the parent class for all transcription exceptions, allowing
    for broad exception handling when needed.
    
    Example:
        try:
            provider.transcribe(audio_path)
        except TranscriptionError as e:
            logger.error(f"Transcription failed: {e}")
    """
    pass


class ConfigurationError(TranscriptionError):
    """Exception raised for transcription configuration errors.
    
    This exception is raised when:
    - Required configuration values are missing
    - Configuration values are invalid
    - Configuration file cannot be loaded
    - Environment variables are not set correctly
    
    Example:
        if not config.api_key:
            raise ConfigurationError("API key not configured")
    """
    pass


class ProviderError(TranscriptionError):
    """Exception raised for transcription provider errors.
    
    This exception is raised when:
    - Provider fails to initialize
    - Provider encounters an error during transcription
    - Provider-specific operations fail
    - API calls fail
    
    Example:
        try:
            response = api.transcribe(audio)
        except Exception as e:
            raise ProviderError(f"Transcription failed: {e}")
    """
    pass


class ProviderNotAvailableError(TranscriptionError):
    """Exception raised when a transcription provider is not available.
    
    This exception is raised when:
    - Required dependencies are not installed
    - API credentials are missing or invalid
    - Service is not accessible (network issues, service down)
    - Provider validation fails
    
    Example:
        if not provider.validate_requirements():
            raise ProviderNotAvailableError("Provider requirements not met")
    """
    pass


class AudioFileError(TranscriptionError):
    """Exception raised for audio file errors.
    
    This exception is raised when:
    - Audio file doesn't exist
    - Audio file format is not supported
    - Audio file is corrupted or invalid
    - Audio file is too large
    
    Example:
        if not os.path.exists(audio_path):
            raise AudioFileError(f"Audio file not found: {audio_path}")
    """
    pass


class TranscriptionTimeoutError(TranscriptionError):
    """Exception raised when transcription operation times out.
    
    This exception is raised when:
    - Transcription takes longer than configured timeout
    - API request times out
    - Long-running operation exceeds limit
    
    Example:
        if elapsed_time > timeout:
            raise TranscriptionTimeoutError(f"Transcription timed out after {timeout}s")
    """
    pass
