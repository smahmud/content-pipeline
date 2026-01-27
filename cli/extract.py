"""
Extract Subcommand Module

This module implements the extract subcommand for the Content Pipeline CLI.
It handles audio and metadata extraction from various sources including
YouTube URLs, local files, and cloud storage.
"""

import os
import json
import sys
import logging
import click
from pipeline.extractors.youtube.extractor import YouTubeExtractor
from pipeline.extractors.dispatch import classify_source
from pipeline.extractors.schema.metadata import build_local_placeholder_metadata
from pipeline.extractors.local.file_audio import extract_audio_from_file
from .shared_options import input_option, output_option
from .help_texts import EXTRACT_HELP, EXTRACT_SOURCE_HELP, EXTRACT_OUTPUT_HELP


@click.command()
@input_option(help=EXTRACT_SOURCE_HELP)
@output_option(help=EXTRACT_OUTPUT_HELP)
def extract(source, output):
    """
    Extract audio from the source file and save it to the specified output path.
    
    Supports extraction from:
    - YouTube URLs (streaming)
    - Local video files (file_system)
    - Cloud storage URLs (storage) - placeholder for future implementation
    """
    source_type = classify_source(source)
    
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", output)
    metadata_path = output_path.replace(".mp3", ".json")

    if source_type == "streaming":        
        extractor = YouTubeExtractor()
        try:
            metadata = extractor.extract_metadata(source)            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
                logging.error(f"Failed to extract or save metadata: {e}")
                print("Warning: Metadata extraction failed.")

        try:
            extractor.extract_audio(source, output_path)
            logging.info(f"Audio saved to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to extract audio: {e}")
            print("Warning: Audio extraction failed.")
    
    elif source_type == "storage":
        metadata = build_local_placeholder_metadata(source)
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            print("Warning: Could not save metadata.")

        # Placeholder for future extractor logic
        logging.warning("Cloud storage extraction not yet implemented.")

    else:  # file_system
        if not os.path.exists(source):
            logging.error(f"Input file not found: {source}")
            print("Error: Input file does not exist.")
            sys.exit(1)

        metadata = build_local_placeholder_metadata(source)
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            print("Warning: Could not save metadata.")

        try:
            extract_audio_from_file(source, output_path)
            logging.info(f"Audio extracted from local file: {output_path}")
        except Exception as e:
            logging.error(f"Failed to extract audio from local file: {e}")            
            print("Warning: Audio extraction failed.")
        
    print("\n Done. You may continue using the terminal.")