"""
Environment variable integration for Enhanced Transcription & Configuration v0.6.5.

This module provides utilities for working with environment variables in the configuration system.
It centralizes environment variable names and provides validation and documentation.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import os
from typing import Dict, List, Optional, Tuple


class EnvironmentVariables:
    """Centralized environment variable definitions and utilities."""
    
    # Core configuration environment variables
    DEFAULT_ENGINE = "CONTENT_PIPELINE_DEFAULT_ENGINE"
    OUTPUT_DIR = "CONTENT_PIPELINE_OUTPUT_DIR"
    LOG_LEVEL = "CONTENT_PIPELINE_LOG_LEVEL"
    
    # Whisper Local configuration
    WHISPER_LOCAL_MODEL = "WHISPER_LOCAL_MODEL"
    
    # OpenAI Whisper API configuration
    OPENAI_API_KEY = "OPENAI_API_KEY"
    WHISPER_API_MODEL = "WHISPER_API_MODEL"
    
    # AWS Transcribe configuration
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    
    @classmethod
    def get_all_variables(cls) -> List[str]:
        """Get list of all supported environment variables."""
        return [
            cls.DEFAULT_ENGINE,
            cls.OUTPUT_DIR,
            cls.LOG_LEVEL,
            cls.WHISPER_LOCAL_MODEL,
            cls.OPENAI_API_KEY,
            cls.WHISPER_API_MODEL,
            cls.AWS_ACCESS_KEY_ID,
            cls.AWS_SECRET_ACCESS_KEY,
            cls.AWS_DEFAULT_REGION
        ]
    
    @classmethod
    def get_variable_documentation(cls) -> Dict[str, str]:
        """Get documentation for all environment variables."""
        return {
            cls.DEFAULT_ENGINE: "Default transcription engine (whisper-local, whisper-api, aws-transcribe, auto)",
            cls.OUTPUT_DIR: "Default output directory for transcripts",
            cls.LOG_LEVEL: "Default logging level (debug, info, warning, error)",
            cls.WHISPER_LOCAL_MODEL: "Default Whisper model size (tiny, base, small, medium, large)",
            cls.OPENAI_API_KEY: "OpenAI API key for Whisper API access",
            cls.WHISPER_API_MODEL: "OpenAI Whisper API model name (default: whisper-1)",
            cls.AWS_ACCESS_KEY_ID: "AWS access key ID for Transcribe service",
            cls.AWS_SECRET_ACCESS_KEY: "AWS secret access key for Transcribe service",
            cls.AWS_DEFAULT_REGION: "AWS region for Transcribe service (default: us-east-1)"
        }
    
    @classmethod
    def validate_environment_setup(cls) -> Tuple[List[str], List[str]]:
        """
        Validate current environment variable setup.
        
        Returns:
            Tuple of (warnings, errors) - warnings for missing optional vars, errors for invalid values
        """
        warnings = []
        errors = []
        
        # Check for engine-specific requirements
        default_engine = os.environ.get(cls.DEFAULT_ENGINE)
        if default_engine:
            if default_engine not in ['whisper-local', 'whisper-api', 'aws-transcribe', 'auto']:
                errors.append(f"Invalid {cls.DEFAULT_ENGINE}: '{default_engine}'. "
                            f"Valid options: whisper-local, whisper-api, aws-transcribe, auto")
        
        # Check log level if set
        log_level = os.environ.get(cls.LOG_LEVEL)
        if log_level:
            if log_level not in ['debug', 'info', 'warning', 'error']:
                errors.append(f"Invalid {cls.LOG_LEVEL}: '{log_level}'. "
                            f"Valid options: debug, info, warning, error")
        
        # Check Whisper model if set
        whisper_model = os.environ.get(cls.WHISPER_LOCAL_MODEL)
        if whisper_model:
            if whisper_model not in ['tiny', 'base', 'small', 'medium', 'large']:
                errors.append(f"Invalid {cls.WHISPER_LOCAL_MODEL}: '{whisper_model}'. "
                            f"Valid options: tiny, base, small, medium, large")
        
        # Check for API key availability based on default engine
        if default_engine == 'whisper-api':
            if not os.environ.get(cls.OPENAI_API_KEY):
                warnings.append(f"Default engine is 'whisper-api' but {cls.OPENAI_API_KEY} is not set")
        
        if default_engine == 'aws-transcribe':
            if not os.environ.get(cls.AWS_ACCESS_KEY_ID):
                warnings.append(f"Default engine is 'aws-transcribe' but {cls.AWS_ACCESS_KEY_ID} is not set")
            if not os.environ.get(cls.AWS_SECRET_ACCESS_KEY):
                warnings.append(f"Default engine is 'aws-transcribe' but {cls.AWS_SECRET_ACCESS_KEY} is not set")
        
        return warnings, errors
    
    @classmethod
    def get_setup_instructions(cls) -> str:
        """Get setup instructions for environment variables."""
        return """
Environment Variable Setup Instructions:

Core Configuration:
  export CONTENT_PIPELINE_DEFAULT_ENGINE=auto
  export CONTENT_PIPELINE_OUTPUT_DIR=./transcripts
  export CONTENT_PIPELINE_LOG_LEVEL=info

Local Whisper:
  export WHISPER_LOCAL_MODEL=base

OpenAI Whisper API:
  export OPENAI_API_KEY=sk-your-openai-key-here
  export WHISPER_API_MODEL=whisper-1

AWS Transcribe:
  export AWS_ACCESS_KEY_ID=your-access-key-id
  export AWS_SECRET_ACCESS_KEY=your-secret-access-key
  export AWS_DEFAULT_REGION=us-east-1

For persistent setup, add these to your shell profile (~/.bashrc, ~/.zshrc, etc.)
"""
    
    @classmethod
    def check_required_for_engine(cls, engine_type: str) -> List[str]:
        """
        Check if required environment variables are set for specific engine.
        
        Args:
            engine_type: Engine type to check (whisper-local, whisper-api, aws-transcribe)
            
        Returns:
            List of missing required environment variables
        """
        missing = []
        
        if engine_type == 'whisper-api':
            if not os.environ.get(cls.OPENAI_API_KEY):
                missing.append(cls.OPENAI_API_KEY)
        
        elif engine_type == 'aws-transcribe':
            if not os.environ.get(cls.AWS_ACCESS_KEY_ID):
                missing.append(cls.AWS_ACCESS_KEY_ID)
            if not os.environ.get(cls.AWS_SECRET_ACCESS_KEY):
                missing.append(cls.AWS_SECRET_ACCESS_KEY)
        
        # whisper-local doesn't require any environment variables
        
        return missing