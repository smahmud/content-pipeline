"""
Shared CLI Option Decorators

This module provides reusable Click decorators for common CLI options,
ensuring consistency across subcommands and following DRY principles.
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