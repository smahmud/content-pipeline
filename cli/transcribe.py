"""
Transcribe Subcommand Module

This module implements the transcribe subcommand for the Content Pipeline CLI.
Enhanced in v0.6.5 to support:
- Multiple transcription engines (whisper-local, whisper-api, aws-transcribe, auto)
- Configuration management with YAML files and environment variables
- Flexible output path management
- Breaking changes with clear migration guidance

BREAKING CHANGES in v0.6.5:
- --engine flag is now REQUIRED
- Output paths changed from hardcoded ./output/ to configurable directories
- Configuration-driven engine selection and preferences
"""

import os
import sys
import logging
import click
from pathlib import Path
from typing import Optional

# Import new v0.6.5 components
from pipeline.config.manager import ConfigurationManager
from pipeline.config.schema import TranscriptionConfig, EngineType, EngineConfig
from pipeline.transcribers.factory import EngineFactory
from pipeline.transcribers.auto_selector import AutoSelector
from pipeline.output.manager import OutputManager
from pipeline.transcribers.normalize import normalize_transcript_v1
from pipeline.transcribers.persistence import LocalFilePersistence

# Import CLI components
from .shared_options import (
    input_option, language_option, engine_option, model_option, 
    api_key_option, output_dir_option, config_option, log_level_option
)
from .help_texts import (
    TRANSCRIBE_HELP, TRANSCRIBE_SOURCE_HELP, TRANSCRIBE_OUTPUT_HELP, 
    TRANSCRIBE_LANGUAGE_HELP, TRANSCRIBE_ENGINE_HELP, TRANSCRIBE_MODEL_HELP,
    TRANSCRIBE_API_KEY_HELP, TRANSCRIBE_OUTPUT_DIR_HELP, TRANSCRIBE_CONFIG_HELP,
    TRANSCRIBE_LOG_LEVEL_HELP, ExitCodes, handle_breaking_change_error,
    show_breaking_change_error
)


@click.command(help=TRANSCRIBE_HELP)
@input_option(help=TRANSCRIBE_SOURCE_HELP)
@click.option("--output", default=None, help=TRANSCRIBE_OUTPUT_HELP)
@language_option(help=TRANSCRIBE_LANGUAGE_HELP)
@engine_option(help=TRANSCRIBE_ENGINE_HELP)
@model_option(help=TRANSCRIBE_MODEL_HELP)
@api_key_option(help=TRANSCRIBE_API_KEY_HELP)
@output_dir_option(help=TRANSCRIBE_OUTPUT_DIR_HELP)
@config_option(help=TRANSCRIBE_CONFIG_HELP)
@log_level_option(help=TRANSCRIBE_LOG_LEVEL_HELP)
def transcribe(source, output, language, engine, model, api_key, output_dir, config, log_level):
    """
    Transcribe audio content to text using configurable speech recognition engines.
    
    Enhanced in v0.6.5 with multiple engine support, configuration management,
    and flexible output handling.
    
    BREAKING CHANGES:
    - --engine flag is now REQUIRED
    - Output paths are now configurable (no longer hardcoded to ./output/)
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigurationManager()
        
        # Load base configuration from files
        transcription_config = config_manager.load_configuration(config_file=config)
        
        # Override with CLI parameters
        transcription_config = _override_config_with_cli_params(
            transcription_config, engine, model, api_key, output_dir, language
        )
        
        logger.debug(f"Final configuration: {transcription_config}")
        
        # Step 2: Validate source file
        if not os.path.exists(source):
            logger.error(f"Audio file not found: {source}")
            click.echo(f"❌ Error: Audio file does not exist: {source}", err=True)
            click.echo("💡 Please check the file path and try again.", err=True)
            sys.exit(ExitCodes.FILE_NOT_FOUND)
        
        # Step 3: Initialize components
        logger.info("Initializing transcription components...")
        factory = EngineFactory()
        output_manager = OutputManager()
        
        # Step 4: Select and validate engine
        selected_engine = _select_and_validate_engine(
            engine, factory, transcription_config, logger
        )
        
        # Step 5: Resolve output path
        output_path = output_manager.resolve_output_path(
            output_path=output,
            output_dir=output_dir or transcription_config.output_dir,
            input_file_path=source
        )
        
        logger.info(f"Output will be saved to: {output_path}")
        
        # Step 6: Create transcription adapter
        logger.info(f"Creating {selected_engine} adapter...")
        adapter = factory.create_adapter(selected_engine, transcription_config)
        
        # Step 7: Run transcription
        logger.info(f"Starting transcription of {source}...")
        click.echo(f"Transcribing {source} using {selected_engine}...")
        
        raw_transcript = adapter.transcribe(source, language=language)
        transcript = normalize_transcript_v1(raw_transcript, adapter)
        
        # Step 8: Save transcript
        logger.info(f"Saving transcript to {output_path}...")
        strategy = LocalFilePersistence()
        strategy.persist(transcript, str(output_path))
        
        # Step 9: Success message
        click.echo(f"✅ Transcription completed successfully!")
        click.echo(f"📄 Transcript saved to: {output_path}")
        
        # Show cost information if available
        if hasattr(adapter, 'estimate_cost'):
            try:
                cost_info = adapter.estimate_cost(source)
                if cost_info and cost_info.get('estimated_cost', 0) > 0:
                    click.echo(f"💰 Estimated cost: ${cost_info['estimated_cost']:.4f}")
            except Exception as e:
                logger.debug(f"Could not get cost information: {e}")
        
    except click.ClickException as e:
        # Handle Click-specific exceptions (like missing required options)
        if "engine" in str(e).lower() or "required" in str(e).lower():
            # This is likely the missing --engine flag
            show_breaking_change_error("engine_required")
        else:
            # Re-raise other Click exceptions normally
            raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        click.echo(f"❌ File not found: {e}", err=True)
        sys.exit(ExitCodes.FILE_NOT_FOUND)
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        click.echo(f"❌ Permission denied: {e}", err=True)
        click.echo("💡 Check file permissions and try again.", err=True)
        sys.exit(ExitCodes.PERMISSION_ERROR)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        error_message = str(e)
        
        # Provide specific guidance based on error type
        if "engine" in error_message.lower() and ("not available" in error_message.lower() or "requirements" in error_message.lower()):
            click.echo(f"❌ Engine Error: {error_message}", err=True)
            handle_breaking_change_error(e, "Engine validation failed")
        elif "api" in error_message.lower() or "key" in error_message.lower() or "auth" in error_message.lower():
            click.echo(f"❌ Authentication Error: {error_message}", err=True)
            show_breaking_change_error("credentials", error_message)
        elif "config" in error_message.lower() or "yaml" in error_message.lower():
            click.echo(f"❌ Configuration Error: {error_message}", err=True)
            show_breaking_change_error("configuration", error_message)
        elif "output" in error_message.lower() or "directory" in error_message.lower():
            click.echo(f"❌ Output Error: {error_message}", err=True)
            show_breaking_change_error("output_path", error_message)
        else:
            # Generic error with migration summary
            click.echo(f"❌ Transcription failed: {error_message}", err=True)
            click.echo("\n💡 If you're upgrading from v0.6.0, see migration guide below:", err=True)
            show_breaking_change_error("migration_summary")


def _override_config_with_cli_params(
    config: TranscriptionConfig,
    engine: str,
    model: Optional[str],
    api_key: Optional[str],
    output_dir: Optional[str],
    language: Optional[str]
) -> TranscriptionConfig:
    """
    Override configuration with CLI parameters.
    
    CLI parameters take precedence over configuration file settings.
    """
    # Override engine selection
    if engine:
        config.engine = engine  # Direct assignment to engine attribute
    
    # Override engine-specific settings
    if model:
        if config.engine == EngineType.WHISPER_LOCAL.value:
            config.whisper_local.model = model
        elif config.engine == EngineType.WHISPER_API.value:
            config.whisper_api.model = model
    
    if api_key:
        if config.engine == EngineType.WHISPER_API.value:
            config.whisper_api.api_key = api_key
    
    # Override output settings
    if output_dir:
        config.output_dir = output_dir
    
    # Override language
    if language:
        config.language = language
    
    return config


def _select_and_validate_engine(
    engine: str,
    factory: EngineFactory,
    config: TranscriptionConfig,
    logger: logging.Logger
) -> str:
    """
    Select and validate the transcription engine.
    
    Handles auto-selection and requirement validation.
    """
    if engine == 'auto':
        logger.info("Auto-selecting best available engine...")
        auto_selector = AutoSelector(factory, config)
        selected_engine, reason = auto_selector.select_engine()
        logger.info(f"Auto-selected engine: {selected_engine} ({reason})")
        click.echo(f"🤖 Auto-selected engine: {selected_engine}")
        click.echo(f"   Reason: {reason}")
        return selected_engine
    else:
        # Use specified engine (keep hyphens as-is for factory)
        engine_type = engine
        
        # Validate engine requirements
        logger.info(f"Validating {engine} requirements...")
        errors = factory.validate_engine_requirements(engine_type, config)
        
        if errors:
            error_msg = f"Engine {engine} is not available:\n" + "\n".join(f"  • {error}" for error in errors)
            logger.error(error_msg)
            click.echo(f"❌ {error_msg}", err=True)
            
            # Provide specific guidance based on error type
            if "api" in error_msg.lower() or "key" in error_msg.lower() or "credential" in error_msg.lower():
                show_breaking_change_error("credentials", f"Engine {engine} requires authentication")
            else:
                # Suggest alternatives
                click.echo("\n💡 Try one of these alternatives:")
                click.echo("  • --engine auto (automatically select best available)")
                click.echo("  • --engine whisper-local (if you have Whisper installed)")
                click.echo("  • Check configuration and credentials")
                sys.exit(ExitCodes.ENGINE_NOT_AVAILABLE)
        
        logger.info(f"Engine {engine} validated successfully")
        return engine_type