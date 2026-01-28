"""
Configuration pretty printer for Enhanced Transcription & Configuration v0.6.5.

This module provides enhanced YAML formatting with comments, examples, and templates
for configuration files.

Requirements: 11.4, 11.5
"""

from typing import Dict, Any, List, Optional
from dataclasses import asdict
from .schema import TranscriptionConfig, EngineType, LogLevel, WhisperModelSize


class ConfigurationPrettyPrinter:
    """Enhanced pretty printer for configuration files with templates and examples."""
    
    def __init__(self):
        self.indent = "  "
    
    def generate_full_template(self, include_examples: bool = True) -> str:
        """
        Generate comprehensive configuration template with all options and examples.
        
        Args:
            include_examples: Whether to include example values and usage patterns
            
        Returns:
            Complete YAML configuration template with comments
        """
        template = """# Content Pipeline Configuration v0.6.5
# Enhanced Transcription & Configuration
# 
# This file configures transcription engines and output settings for the content-pipeline.
# You can customize these settings based on your needs and available resources.

# =============================================================================
# CORE CONFIGURATION
# =============================================================================

# Default transcription engine to use
# Options: local-whisper, openai-whisper, aws-transcribe, auto
# - local-whisper: Process audio locally (privacy-focused, requires local installation)
# - openai-whisper: Use OpenAI's Whisper API (highest quality, requires API key)
# - aws-transcribe: Use AWS Transcribe service (good quality, requires AWS credentials)
# - auto: Automatically select best available engine
engine: auto

# Output directory for transcripts
# Supports environment variable substitution: ${VARIABLE_NAME}
# Examples:
#   output_dir: ./transcripts
#   output_dir: ${HOME}/Documents/transcripts
#   output_dir: ${CONTENT_PIPELINE_OUTPUT_DIR:-./default-output}
output_dir: ./transcripts

# Logging verbosity level
# Options: debug, info, warning, error
# - debug: Detailed execution information (useful for troubleshooting)
# - info: Standard progress information (recommended for normal use)
# - warning: Only warnings and errors
# - error: Only error messages
log_level: info

# Default language hint for transcription (optional)
# Use ISO 639-1 language codes (e.g., 'en', 'es', 'fr', 'de')
# Leave as null to auto-detect language
language: null

# =============================================================================
# LOCAL WHISPER CONFIGURATION
# =============================================================================

whisper_local:
  # Whisper model size (affects accuracy vs speed trade-off)
  # Options: tiny, base, small, medium, large
  # - tiny: Fastest, least accurate (~39 MB)
  # - base: Good balance (~74 MB) - RECOMMENDED
  # - small: Better accuracy (~244 MB)
  # - medium: High accuracy (~769 MB)
  # - large: Highest accuracy (~1550 MB)
  model: base
  
  # Processing device
  # Options: auto, cpu, cuda
  # - auto: Automatically detect best available device
  # - cpu: Force CPU processing (slower but always available)
  # - cuda: Force GPU processing (faster, requires NVIDIA GPU)
  device: auto
  
  # Compute type for faster-whisper (if available)
  # Options: default, int8, int8_float16, int16, float16, float32
  compute_type: default
  
  # Maximum processing time in seconds
  timeout: 300
  
  # Number of retry attempts on failure
  retry_attempts: 3
  
  # Delay between retry attempts in seconds
  retry_delay: 1.0

# =============================================================================
# OPENAI WHISPER API CONFIGURATION
# =============================================================================

whisper_api:
  # OpenAI API key (required for whisper-api engine)
  # Get your API key from: https://platform.openai.com/api-keys
  # SECURITY: Use environment variables instead of hardcoding keys
  api_key: ${OPENAI_API_KEY}
  
  # OpenAI Whisper model name
  # Currently only 'whisper-1' is available
  model: whisper-1
  
  # Sampling temperature (0.0 = deterministic, 1.0 = creative)
  # Range: 0.0 to 1.0
  # Recommended: 0.0 for transcription accuracy
  temperature: 0.0
  
  # Response format
  # Options: json, text, srt, verbose_json, vtt
  response_format: json
  
  # API request timeout in seconds
  timeout: 60
  
  # Number of retry attempts on API failure
  retry_attempts: 3
  
  # Delay between API retry attempts in seconds
  retry_delay: 1.0

# =============================================================================
# AWS TRANSCRIBE CONFIGURATION
# =============================================================================

aws_transcribe:
  # AWS credentials (required for aws-transcribe engine)
  # Get credentials from AWS IAM console
  # SECURITY: Use environment variables or AWS credential files
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  
  # AWS region for Transcribe service
  # Examples: us-east-1, us-west-2, eu-west-1, ap-southeast-1
  region: us-east-1
  
  # Language code for transcription
  # Examples: en-US, es-ES, fr-FR, de-DE, ja-JP, zh-CN
  language_code: en-US
  
  # Media format (auto-detection recommended)
  # Options: auto, mp3, mp4, wav, flac, ogg, amr, webm
  media_format: auto
  
  # Service request timeout in seconds
  timeout: 300
  
  # Number of retry attempts on service failure
  retry_attempts: 3
  
  # Delay between service retry attempts in seconds
  retry_delay: 1.0

# =============================================================================
# AUTO-SELECTION PREFERENCES
# =============================================================================

auto_selection:
  # Prefer local processing when available (privacy-focused)
  prefer_local: true
  
  # Enable fallback to other engines if primary choice fails
  fallback_enabled: true
  
  # Engine priority order for auto-selection
  # Engines are tried in this order until one succeeds
  # Options: local-whisper, openai-whisper, aws-transcribe
  priority_order:
    - local-whisper    # Try local first (privacy + no cost)
    - aws-transcribe   # Then AWS (good quality, user has credits)
    - openai-whisper      # Finally OpenAI (highest quality, but paid)

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Basic usage with explicit engine:
#   content-pipeline transcribe --engine local-whisper --source audio.mp3
#   content-pipeline transcribe --engine openai-whisper --source audio.mp3
#   content-pipeline transcribe --engine auto --source audio.mp3
#
# With custom output:
#   content-pipeline transcribe --engine auto --source audio.mp3 --output ./my-transcript.json
#   content-pipeline transcribe --engine auto --source audio.mp3 --output-dir ./my-transcripts/
#
# With configuration file:
#   content-pipeline transcribe --engine auto --source audio.mp3 --config ./my-config.yaml
#
# Environment variable examples:
#   export CONTENT_PIPELINE_DEFAULT_ENGINE=whisper-local
#   export CONTENT_PIPELINE_OUTPUT_DIR=./my-default-output
#   export OPENAI_API_KEY=sk-your-openai-key-here
#   export AWS_ACCESS_KEY_ID=your-aws-access-key
#   export AWS_SECRET_ACCESS_KEY=your-aws-secret-key
#
# =============================================================================
"""
        
        if not include_examples:
            # Remove the usage examples section
            lines = template.split('\n')
            example_start = None
            for i, line in enumerate(lines):
                if 'USAGE EXAMPLES' in line:
                    example_start = i - 2  # Include the separator line
                    break
            
            if example_start:
                template = '\n'.join(lines[:example_start])
        
        return template
    
    def generate_minimal_template(self) -> str:
        """Generate minimal configuration template with only essential settings."""
        return """# Content Pipeline Configuration v0.6.5
# Minimal configuration template

# Transcription engine (local-whisper, openai-whisper, aws-transcribe, auto)
engine: auto

# Output directory
output_dir: ./transcripts

# Logging level (debug, info, warning, error)
log_level: info

# Local Whisper settings
whisper_local:
  model: base

# OpenAI API settings (if using whisper-api)
whisper_api:
  api_key: ${OPENAI_API_KEY}

# AWS settings (if using aws-transcribe)
aws_transcribe:
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  region: us-east-1
"""
    
    def generate_engine_specific_template(self, engine_type: str) -> str:
        """
        Generate configuration template optimized for specific engine.
        
        Args:
            engine_type: Target engine type (local-whisper, openai-whisper, aws-transcribe)
            
        Returns:
            Engine-specific configuration template
        """
        if engine_type == EngineType.WHISPER_LOCAL.value:
            return self._generate_whisper_local_template()
        elif engine_type == EngineType.WHISPER_API.value:
            return self._generate_whisper_api_template()
        elif engine_type == EngineType.AWS_TRANSCRIBE.value:
            return self._generate_aws_transcribe_template()
        else:
            return self.generate_minimal_template()
    
    def _generate_whisper_local_template(self) -> str:
        """Generate template optimized for local Whisper usage."""
        return """# Content Pipeline Configuration v0.6.5
# Optimized for Local Whisper Processing

# Use local Whisper for privacy and offline processing
engine: local-whisper

# Output directory
output_dir: ./transcripts

# Logging level
log_level: info

# Local Whisper configuration
whisper_local:
  # Model size (tiny, base, small, medium, large)
  # Larger models are more accurate but slower
  model: base
  
  # Processing device (auto, cpu, cuda)
  # Use 'cuda' if you have an NVIDIA GPU for faster processing
  device: auto
  
  # Processing timeout in seconds
  timeout: 300

# Auto-selection preferences (fallback options)
auto_selection:
  prefer_local: true
  fallback_enabled: false  # Disable fallback to stay local-only
"""
    
    def _generate_whisper_api_template(self) -> str:
        """Generate template optimized for OpenAI Whisper API usage."""
        return """# Content Pipeline Configuration v0.6.5
# Optimized for OpenAI Whisper API

# Use OpenAI Whisper API for highest quality transcription
engine: openai-whisper

# Output directory
output_dir: ./transcripts

# Logging level
log_level: info

# OpenAI Whisper API configuration
whisper_api:
  # API key (get from https://platform.openai.com/api-keys)
  api_key: ${OPENAI_API_KEY}
  
  # Model (currently only whisper-1 available)
  model: whisper-1
  
  # Temperature for sampling (0.0 = deterministic)
  temperature: 0.0
  
  # API timeout in seconds
  timeout: 60

# Auto-selection preferences
auto_selection:
  prefer_local: false
  fallback_enabled: true
  priority_order:
    - openai-whisper
    - local-whisper  # Fallback to local if API fails
"""
    
    def _generate_aws_transcribe_template(self) -> str:
        """Generate template optimized for AWS Transcribe usage."""
        return """# Content Pipeline Configuration v0.6.5
# Optimized for AWS Transcribe Service

# Use AWS Transcribe for cloud processing with your AWS credits
engine: aws-transcribe

# Output directory
output_dir: ./transcripts

# Logging level
log_level: info

# AWS Transcribe configuration
aws_transcribe:
  # AWS credentials
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  
  # AWS region
  region: us-east-1
  
  # Language code
  language_code: en-US
  
  # Media format (auto-detect recommended)
  media_format: auto
  
  # Service timeout in seconds
  timeout: 300

# Auto-selection preferences
auto_selection:
  prefer_local: false
  fallback_enabled: true
  priority_order:
    - aws-transcribe
    - local-whisper  # Fallback to local if AWS fails
"""
    
    def format_configuration(self, config: TranscriptionConfig, 
                           style: str = "full") -> str:
        """
        Format configuration object as pretty YAML.
        
        Args:
            config: Configuration object to format
            style: Formatting style ("full", "minimal", "compact")
            
        Returns:
            Formatted YAML string
        """
        if style == "full":
            return self._format_full_configuration(config)
        elif style == "minimal":
            return self._format_minimal_configuration(config)
        elif style == "compact":
            return self._format_compact_configuration(config)
        else:
            return self._format_full_configuration(config)
    
    def _format_full_configuration(self, config: TranscriptionConfig) -> str:
        """Format configuration with full comments and explanations."""
        config_dict = asdict(config)
        
        return f"""# Content Pipeline Configuration v0.6.5
# Generated from current settings

# Transcription engine
engine: {config_dict['engine']}

# Output directory
output_dir: {config_dict['output_dir']}

# Logging level
log_level: {config_dict['log_level']}

# Language hint
language: {config_dict['language']}

# Local Whisper settings
whisper_local:
  model: {config_dict['whisper_local']['model']}
  device: {config_dict['whisper_local']['device']}
  compute_type: {config_dict['whisper_local']['compute_type']}
  timeout: {config_dict['whisper_local']['timeout']}
  retry_attempts: {config_dict['whisper_local']['retry_attempts']}
  retry_delay: {config_dict['whisper_local']['retry_delay']}

# OpenAI API settings
whisper_api:
  api_key: {config_dict['whisper_api']['api_key'] or '${OPENAI_API_KEY}'}
  model: {config_dict['whisper_api']['model']}
  temperature: {config_dict['whisper_api']['temperature']}
  response_format: {config_dict['whisper_api']['response_format']}
  timeout: {config_dict['whisper_api']['timeout']}
  retry_attempts: {config_dict['whisper_api']['retry_attempts']}
  retry_delay: {config_dict['whisper_api']['retry_delay']}

# AWS Transcribe settings
aws_transcribe:
  access_key_id: {config_dict['aws_transcribe']['access_key_id'] or '${AWS_ACCESS_KEY_ID}'}
  secret_access_key: {config_dict['aws_transcribe']['secret_access_key'] or '${AWS_SECRET_ACCESS_KEY}'}
  region: {config_dict['aws_transcribe']['region']}
  language_code: {config_dict['aws_transcribe']['language_code']}
  media_format: {config_dict['aws_transcribe']['media_format']}
  timeout: {config_dict['aws_transcribe']['timeout']}
  retry_attempts: {config_dict['aws_transcribe']['retry_attempts']}
  retry_delay: {config_dict['aws_transcribe']['retry_delay']}

# Auto-selection preferences
auto_selection:
  prefer_local: {str(config_dict['auto_selection']['prefer_local']).lower()}
  fallback_enabled: {str(config_dict['auto_selection']['fallback_enabled']).lower()}
  priority_order:
{chr(10).join(f'    - {engine}' for engine in config_dict['auto_selection']['priority_order'])}
"""
    
    def _format_minimal_configuration(self, config: TranscriptionConfig) -> str:
        """Format configuration with minimal comments."""
        config_dict = asdict(config)
        
        return f"""# Content Pipeline Configuration v0.6.5
engine: {config_dict['engine']}
output_dir: {config_dict['output_dir']}
log_level: {config_dict['log_level']}
language: {config_dict['language']}

whisper_local:
  model: {config_dict['whisper_local']['model']}
  device: {config_dict['whisper_local']['device']}

whisper_api:
  api_key: {config_dict['whisper_api']['api_key'] or '${OPENAI_API_KEY}'}
  model: {config_dict['whisper_api']['model']}

aws_transcribe:
  access_key_id: {config_dict['aws_transcribe']['access_key_id'] or '${AWS_ACCESS_KEY_ID}'}
  secret_access_key: {config_dict['aws_transcribe']['secret_access_key'] or '${AWS_SECRET_ACCESS_KEY}'}
  region: {config_dict['aws_transcribe']['region']}
"""
    
    def _format_compact_configuration(self, config: TranscriptionConfig) -> str:
        """Format configuration in compact form with no comments."""
        config_dict = asdict(config)
        
        # Only include non-default values
        compact_dict = {}
        defaults = asdict(TranscriptionConfig())
        
        for key, value in config_dict.items():
            if value != defaults[key]:
                compact_dict[key] = value
        
        import yaml
        return yaml.dump(compact_dict, default_flow_style=False, sort_keys=False)