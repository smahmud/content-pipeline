"""
CLI Package for Content Pipeline

This package provides a modular CLI architecture using Click groups and subcommands.
Each subcommand is implemented in its own module for better maintainability and testing.

The main entry point is the main() function which creates a Click group and registers
all available subcommands. The cli() function serves as the console script entry point
for setup.py.
"""

import click
from pipeline.config.logging_config import configure_logging
from .extract import extract
from .transcribe import transcribe
from .enrich import enrich

# Configure logging when CLI package is imported
configure_logging()

@click.group()
@click.version_option(version='0.7.0', prog_name='content-pipeline')
def main():
    """Content Pipeline CLI - Extract, transcribe, and enrich multimedia content.
    
    A modular pipeline for extracting, enriching, and publishing audio-based 
    content from platforms like YouTube. Built for transparency, auditability, 
    and enterprise-grade scalability.
    """
    pass

# Register subcommands
main.add_command(extract)
main.add_command(transcribe)
main.add_command(enrich)

# Entry point for setup.py console script
def cli():
    """Console script entry point.
    
    This function is called when the content-pipeline command is executed
    from the command line after installation via pip.
    """
    main()