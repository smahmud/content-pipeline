"""
Centralized Help Text Constants

This module provides all CLI help text constants for commands and options,
ensuring consistency across subcommands and enabling easy maintenance.
"""

# Command help texts
EXTRACT_HELP = "Extract audio/video content from various sources including URLs, files, and cloud storage."
TRANSCRIBE_HELP = "Transcribe audio content to text using speech recognition."

# Option help texts - Extract command
EXTRACT_SOURCE_HELP = (
    "Streaming platform URL (currently only YouTube) or a YouTube video file (.mp4) in the local file system. "
    "Future support includes Vimeo, TikTok, and cloud-hosted video files."
)

EXTRACT_OUTPUT_HELP = (
    "Base filename for extracted audio (.mp3) and its metadata (.json). "
    "Currently saved to the local file system; future support includes cloud destinations."
)

# Option help texts - Transcribe command
TRANSCRIBE_SOURCE_HELP = (
    "Path to an audio file (.mp3) in the local file system. "
    "Future support includes cloud-hosted audio files."
)

TRANSCRIBE_OUTPUT_HELP = (
    "Base filename for transcript (.json) generated from audio. Uses TranscriptV1 schema. "
    "Currently saved to the local file system; future support includes cloud destinations."
)

TRANSCRIBE_LANGUAGE_HELP = (
    "Optional language hint for transcription (e.g., 'en', 'fr'). "
    "Improves accuracy when language is known. If omitted, language will be auto-detected."
)

# Generic option help texts
SOURCE_HELP = "Input source: URL (YouTube, etc.), local file path, or cloud storage location"
OUTPUT_HELP = "Output file path where extracted/transcribed content will be saved"
LANGUAGE_HELP = "Language hint to improve transcription accuracy (e.g., 'en', 'es', 'fr')"
FORMAT_HELP = "Output format for extracted content"

# Error messages
MISSING_SOURCE_ERROR = "Source file or URL is required"
INVALID_FORMAT_ERROR = "Unsupported format specified"
FILE_NOT_FOUND_ERROR = "Source file not found: {path}"

