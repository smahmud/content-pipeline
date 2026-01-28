"""
Shared CLI Option Decorators

This module provides reusable Click decorators for common CLI options,
ensuring consistency across subcommands and following DRY principles.

Enhanced in v0.6.5 to support:
- Engine selection with validation
- Configuration file options
- Output directory management
- Logging level control
"""

import click
from functools import wraps

def input_option(help=None):
    """Decorator for source/input file options."""
    def decorator(f):
        return click.option(
            '--source', '-s',
            required=True,
            help=help or 'Input source (URL, file path, or storage location)'
        )(f)
    return decorator

def output_option(help=None):
    """Decorator for output file options."""
    def decorator(f):
        return click.option(
            '--output', '-o',
            default="output.mp3",
            help=help or 'Output file path'
        )(f)
    return decorator

def language_option(help=None):
    """Decorator for language hint options."""
    def decorator(f):
        return click.option(
            '--language', '-l',
            default=None,
            help=help or 'Language hint for processing'
        )(f)
    return decorator

def engine_option(help=None):
    """Decorator for transcription engine selection."""
    def decorator(f):
        return click.option(
            '--engine', '-e',
            required=True,
            type=click.Choice(['whisper-local', 'whisper-api', 'aws-transcribe', 'auto'], case_sensitive=False),
            help=help or 'Transcription engine to use'
        )(f)
    return decorator

def model_option(help=None):
    """Decorator for model selection."""
    def decorator(f):
        return click.option(
            '--model', '-m',
            default=None,
            help=help or 'Model to use for transcription (engine-specific)'
        )(f)
    return decorator

def api_key_option(help=None):
    """Decorator for API key options."""
    def decorator(f):
        return click.option(
            '--api-key',
            default=None,
            help=help or 'API key for cloud transcription services'
        )(f)
    return decorator

def output_dir_option(help=None):
    """Decorator for output directory options."""
    def decorator(f):
        return click.option(
            '--output-dir',
            default=None,
            help=help or 'Directory for output files (overrides config)'
        )(f)
    return decorator

def config_option(help=None):
    """Decorator for configuration file options."""
    def decorator(f):
        return click.option(
            '--config',
            default=None,
            help=help or 'Path to configuration file'
        )(f)
    return decorator

def log_level_option(help=None):
    """Decorator for logging level options."""
    def decorator(f):
        return click.option(
            '--log-level',
            default='INFO',
            type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
            help=help or 'Logging level'
        )(f)
    return decorator