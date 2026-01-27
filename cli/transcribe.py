"""
Transcribe Subcommand Module

This module implements the transcribe subcommand for the Content Pipeline CLI.
It handles audio transcription using various speech recognition engines,
with normalization and structured output.
"""

import os
import sys
import logging
import click
from pipeline.transcribers.adapters.whisper import WhisperAdapter
from pipeline.transcribers.normalize import normalize_transcript_v1
from pipeline.transcribers.persistence import LocalFilePersistence
from .shared_options import input_option, output_option, language_option
from .help_texts import TRANSCRIBE_HELP, TRANSCRIBE_SOURCE_HELP, TRANSCRIBE_OUTPUT_HELP, TRANSCRIBE_LANGUAGE_HELP


@click.command()
@input_option(help=TRANSCRIBE_SOURCE_HELP)
@click.option("--output", default="transcript.json", help=TRANSCRIBE_OUTPUT_HELP)
@language_option(help=TRANSCRIBE_LANGUAGE_HELP)
def transcribe(source, output, language):
    """
    Extract audio from the source, run transcription, and save the normalized transcript.
    
    Uses OpenAI Whisper for speech recognition with optional language hints
    for improved accuracy. Output follows the TranscriptV1 schema.
    """
    # Validate source file
    if not os.path.exists(source):
        logging.error(f"Audio file not found: {source}")
        print("Error: Audio file does not exist.")
        sys.exit(1)

    # Prepare output paths
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", output)

    # Run transcription
    adapter = WhisperAdapter(model_name="base")  # You can make model configurable later
    raw_transcript = adapter.transcribe(source, language=language)
    transcript = normalize_transcript_v1(raw_transcript, adapter)

    # Save transcript
    try:
        strategy = LocalFilePersistence()
        strategy.persist(transcript, output_path)
        logging.info(f"Transcript saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save transcript: {e}")
        print("Warning: Could not save transcript.")

    print("\n Done. Transcript generated.")