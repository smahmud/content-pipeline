"""
Centralized Help Text Constants

This module provides all CLI help text constants for commands and options,
ensuring consistency across subcommands and enabling easy maintenance.

Enhanced in v0.6.5 to support:
- Engine selection help and migration guidance
- Configuration management help
- Breaking change error messages
- Comprehensive error handling with appropriate exit codes
"""

# Exit codes for different error types
class ExitCodes:
    SUCCESS = 0
    GENERAL_ERROR = 1
    MISSING_REQUIRED_OPTION = 2
    INVALID_CONFIGURATION = 3
    ENGINE_NOT_AVAILABLE = 4
    AUTHENTICATION_ERROR = 5
    FILE_NOT_FOUND = 6
    PERMISSION_ERROR = 7
    NETWORK_ERROR = 8
    BREAKING_CHANGE_ERROR = 10

# Command help texts
EXTRACT_HELP = "Extract audio/video content from various sources including URLs, files, and cloud storage."
TRANSCRIBE_HELP = "Transcribe audio content to text using configurable speech recognition engines."

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
    "Specific output file path for transcript (.json). If not specified, uses output-dir + input filename."
)

TRANSCRIBE_LANGUAGE_HELP = (
    "Optional language hint for transcription (e.g., 'en', 'fr'). "
    "Improves accuracy when language is known. If omitted, language will be auto-detected."
)

# New option help texts for v0.6.5
TRANSCRIBE_ENGINE_HELP = (
    "Transcription engine to use:\n"
    "  local-whisper: Local Whisper model (privacy-focused, no internet required)\n"
    "  openai-whisper: OpenAI Whisper API (high quality, requires API key and credits)\n"
    "  aws-transcribe: AWS Transcribe service (good quality, requires AWS credentials)\n"
    "  auto: Automatically select best available engine based on configuration"
)

TRANSCRIBE_MODEL_HELP = (
    "Model to use for transcription. Options vary by engine:\n"
    "  local-whisper: tiny, base, small, medium, large (default: base)\n"
    "  openai-whisper: whisper-1 (default and only option)\n"
    "  aws-transcribe: Uses service defaults\n"
    "  auto: Uses configured model for selected engine"
)

TRANSCRIBE_API_KEY_HELP = (
    "API key for cloud transcription services. Required for whisper-api. "
    "Can also be set via OPENAI_API_KEY environment variable."
)

TRANSCRIBE_OUTPUT_DIR_HELP = (
    "Directory for output files. Overrides configuration file settings. "
    "If not specified, uses configured output directory or './transcripts'."
)

TRANSCRIBE_CONFIG_HELP = (
    "Path to configuration file (.yaml). If not specified, looks for:\n"
    "  1. ./.content-pipeline/config.yaml (project config)\n"
    "  2. ~/.content-pipeline/config.yaml (user config)\n"
    "  3. Built-in defaults"
)

TRANSCRIBE_LOG_LEVEL_HELP = (
    "Logging level for detailed output. Use DEBUG for troubleshooting."
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

# Breaking change migration messages
BREAKING_CHANGE_ENGINE_REQUIRED = """
BREAKING CHANGE in v0.6.5

The --engine flag is now REQUIRED for transcription.

Previous usage (no longer works):
  content-pipeline transcribe --source audio.mp3

New usage (required):
  content-pipeline transcribe --source audio.mp3 --engine local-whisper

Available engines:
  --engine local-whisper    # Local Whisper (privacy-focused, no internet)
  --engine openai-whisper      # OpenAI API (high quality, requires API key)
  --engine aws-transcribe   # AWS Transcribe (requires AWS credentials)
  --engine auto             # Automatically select best available

For the same behavior as before, use:
  content-pipeline transcribe --source audio.mp3 --engine local-whisper --model base

See documentation for configuration file setup to avoid repeating options.
"""

BREAKING_CHANGE_OUTPUT_PATH = """
OUTPUT PATH CHANGE in v0.6.5

Output files are no longer saved to hardcoded './output/' directory.

Previous behavior:
  Files saved to: ./output/transcript.json

New behavior:
  Default: ./transcripts/[input-filename].json
  Custom: Use --output for specific file or --output-dir for directory

Examples:
  # Use default directory
  content-pipeline transcribe --source audio.mp3 --engine local-whisper
  # Result: ./transcripts/audio.json

  # Specify output directory
  content-pipeline transcribe --source audio.mp3 --engine local-whisper --output-dir ./my-transcripts
  # Result: ./my-transcripts/audio.json

  # Specify exact output file
  content-pipeline transcribe --source audio.mp3 --engine local-whisper --output ./results/my-transcript.json
  # Result: ./results/my-transcript.json
"""

BREAKING_CHANGE_CONFIGURATION = """
CONFIGURATION CHANGE in v0.6.5

Configuration is now managed through YAML files and environment variables.

New configuration locations (checked in order):
  1. --config /path/to/config.yaml (CLI override)
  2. ./.content-pipeline/config.yaml (project config)
  3. ~/.content-pipeline/config.yaml (user config)
  4. Built-in defaults

Example configuration file (~/.content-pipeline/config.yaml):
  engine: local-whisper
  output_dir: ./transcripts
  whisper_local:
    model: base
  whisper_api:
    api_key: ${OPENAI_API_KEY}
  aws_transcribe:
    region: us-east-1

Environment variables (override config files):
  OPENAI_API_KEY=your-key-here
  CONTENT_PIPELINE_DEFAULT_ENGINE=whisper-local
  CONTENT_PIPELINE_OUTPUT_DIR=./my-transcripts
  CONTENT_PIPELINE_LOG_LEVEL=INFO

This eliminates the need to specify the same options repeatedly.
"""

BREAKING_CHANGE_API_CREDENTIALS = """
API CREDENTIAL SETUP REQUIRED

Cloud transcription engines require proper authentication setup.

For OpenAI Whisper API (--engine openai-whisper):
  Option 1: Environment variable
    export OPENAI_API_KEY="your-api-key-here"
  
  Option 2: CLI flag
    --api-key your-api-key-here
  
  Option 3: Configuration file
    whisper_api:
      api_key: ${OPENAI_API_KEY}

For AWS Transcribe (--engine aws-transcribe):
  Option 1: AWS CLI configuration
    aws configure
  
  Option 2: Environment variables
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_DEFAULT_REGION="us-east-1"
  
  Option 3: IAM roles (for EC2/Lambda)

Get API keys:
  • OpenAI: https://platform.openai.com/api-keys
  • AWS: https://console.aws.amazon.com/iam/
"""

BREAKING_CHANGE_MIGRATION_SUMMARY = """
MIGRATION SUMMARY: v0.6.0 -> v0.6.5

Key changes you need to know:

1. ENGINE SELECTION NOW REQUIRED
   Old: content-pipeline transcribe --source audio.mp3
   New: content-pipeline transcribe --source audio.mp3 --engine local-whisper

2. OUTPUT PATHS CHANGED
   Old: Files saved to ./output/
   New: Files saved to ./transcripts/ (configurable)

3. CONFIGURATION FILES SUPPORTED
   Create ~/.content-pipeline/config.yaml to set defaults:
     engine: local-whisper
     output_dir: ./transcripts
     whisper_local:
       model: base

4. NEW ENGINE OPTIONS
   • local-whisper: Same as before (local processing)
   • openai-whisper: OpenAI API (requires API key)
   • aws-transcribe: AWS service (requires AWS credentials)
   • auto: Automatically select best available

5. ENVIRONMENT VARIABLES
   Set OPENAI_API_KEY, AWS credentials, or CONTENT_PIPELINE_* variables

Quick migration for common patterns:
  # Basic transcription (same as v0.6.0)
  content-pipeline transcribe --source audio.mp3 --engine local-whisper
  
  # High quality with OpenAI
  export OPENAI_API_KEY="your-key"
  content-pipeline transcribe --source audio.mp3 --engine openai-whisper
  
  # Let system choose best option
  content-pipeline transcribe --source audio.mp3 --engine auto
"""


def show_breaking_change_error(error_type: str, additional_context: str = None):
    """
    Display appropriate breaking change error message based on error type.
    
    Args:
        error_type: Type of breaking change error
        additional_context: Additional context-specific information
    """
    import click
    import sys
    
    if error_type == "engine_required":
        click.echo(BREAKING_CHANGE_ENGINE_REQUIRED, err=True)
        sys.exit(ExitCodes.BREAKING_CHANGE_ERROR)
    
    elif error_type == "output_path":
        click.echo(BREAKING_CHANGE_OUTPUT_PATH, err=True)
        if additional_context:
            click.echo(f"\nAdditional context: {additional_context}", err=True)
        sys.exit(ExitCodes.BREAKING_CHANGE_ERROR)
    
    elif error_type == "configuration":
        click.echo(BREAKING_CHANGE_CONFIGURATION, err=True)
        if additional_context:
            click.echo(f"\nAdditional context: {additional_context}", err=True)
        sys.exit(ExitCodes.INVALID_CONFIGURATION)
    
    elif error_type == "credentials":
        click.echo(BREAKING_CHANGE_API_CREDENTIALS, err=True)
        if additional_context:
            click.echo(f"\nAdditional context: {additional_context}", err=True)
        sys.exit(ExitCodes.AUTHENTICATION_ERROR)
    
    elif error_type == "migration_summary":
        click.echo(BREAKING_CHANGE_MIGRATION_SUMMARY, err=True)
        sys.exit(ExitCodes.BREAKING_CHANGE_ERROR)
    
    else:
        # Default to general migration summary
        click.echo(BREAKING_CHANGE_MIGRATION_SUMMARY, err=True)
        sys.exit(ExitCodes.BREAKING_CHANGE_ERROR)


def suggest_migration_for_error(error_message: str) -> str:
    """
    Analyze error message and suggest appropriate migration guidance.
    
    Args:
        error_message: The error message to analyze
        
    Returns:
        Suggested migration message type
    """
    error_lower = error_message.lower()
    
    if "engine" in error_lower and ("required" in error_lower or "missing" in error_lower):
        return "engine_required"
    
    elif "output" in error_lower or "directory" in error_lower or "path" in error_lower:
        return "output_path"
    
    elif "config" in error_lower or "yaml" in error_lower:
        return "configuration"
    
    elif "api" in error_lower or "key" in error_lower or "auth" in error_lower or "credential" in error_lower:
        return "credentials"
    
    else:
        return "migration_summary"


def handle_breaking_change_error(error: Exception, context: str = None):
    """
    Handle breaking change errors with appropriate migration guidance.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    error_message = str(error)
    migration_type = suggest_migration_for_error(error_message)
    
    additional_context = context
    if context and error_message:
        additional_context = f"{context}: {error_message}"
    elif error_message:
        additional_context = error_message
    
    show_breaking_change_error(migration_type, additional_context)

