"""
CLI Package for Content Pipeline

This package provides a modular CLI architecture using Click groups and subcommands.
Each subcommand is implemented in its own module for better maintainability and testing.

The main entry point is the main() function which creates a Click group and registers
all available subcommands. The cli() function serves as the console script entry point
for setup.py.
"""

import os
import click
from dotenv import load_dotenv
from pipeline.config.logging_config import configure_logging

# Load environment variables from .env files
# Priority: .env.dev (if exists) overrides .env
if os.path.exists('.env.dev'):
    load_dotenv('.env.dev')
elif os.path.exists('.env'):
    load_dotenv('.env')
from .extract import extract
from .transcribe import transcribe
from .enrich import enrich
from .format import format
from .validate import validate

# Configure logging when CLI package is imported
configure_logging()

@click.group()
@click.version_option(version='1.0.0', prog_name='content-pipeline')
def main():
    """Content Pipeline CLI - Extract, transcribe, enrich, and format multimedia content.
    
    A modular pipeline for extracting, enriching, formatting, and publishing audio-based 
    content from platforms like YouTube. Built for transparency, auditability, 
    and enterprise-grade scalability.
    """
    pass

# Register subcommands
main.add_command(extract)
main.add_command(transcribe)
main.add_command(enrich)
main.add_command(format)
main.add_command(validate)

# Entry point for setup.py console script
def cli():
    """Console script entry point.
    
    This function is called when the content-pipeline command is executed
    from the command line after installation via pip.
    """
    main()