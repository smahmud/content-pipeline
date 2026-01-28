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
import time
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
from pipeline.utils.logging_config import logging_config
from pipeline.utils.error_messages import ErrorMessages, ErrorCategory

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
    # Configure logging first
    logging_config.configure_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info(f"Content Pipeline v0.6.5 - Transcribe Command")
    logger.debug(f"CLI arguments: source={source}, engine={engine}, model={model}, log_level={log_level}")
    
    try:
        # Step 1: Load configuration
        with logging_config.progress_context("Loading configuration") as progress:
            logger.info("Loading configuration...")
            config_manager = ConfigurationManager()
            
            # Load base configuration from files
            transcription_config = config_manager.load_configuration(config_file=config)
            progress.update(message="Configuration files loaded")
            
            # Override with CLI parameters
            transcription_config = _override_config_with_cli_params(
                transcription_config, engine, model, api_key, output_dir, language
            )
            progress.update(message="CLI overrides applied")
            
            # Log configuration details at debug level
            if logger.isEnabledFor(logging.DEBUG):
                config_dict = {
                    'engine': transcription_config.engine,
                    'output_dir': transcription_config.output_dir,
                    'model': getattr(transcription_config, 'model', 'default'),
                    'language': transcription_config.language,
                    'log_level': log_level
                }
                logging_config.log_configuration_details(config_dict)
        
        # Step 2: Validate source file
        logger.debug(f"Validating source file: {source}")
        if not os.path.exists(source):
            error_msg = ErrorMessages.format_error(
                ErrorCategory.FILE_ACCESS,
                "file_not_found",
                file_path=source
            )
            logger.error(f"Audio file not found: {source}")
            click.echo(error_msg, err=True)
            sys.exit(ExitCodes.FILE_NOT_FOUND)
        
        # Step 3: Initialize components
        with logging_config.progress_context("Initializing components") as progress:
            logger.info("Initializing transcription components...")
            factory = EngineFactory()
            output_manager = OutputManager()
            progress.update(message="Components initialized")
        
        # Step 4: Select and validate engine
        with logging_config.progress_context("Validating engine") as progress:
            selected_engine = _select_and_validate_engine(
                engine, factory, transcription_config, logger
            )
            progress.update(message=f"Engine {selected_engine} validated")
        
        # Step 5: Resolve output path
        logger.debug("Resolving output path...")
        output_path = output_manager.resolve_output_path(
            output_path=output,
            output_dir=output_dir or transcription_config.output_dir,
            input_file_path=source
        )
        
        logger.info(f"Output will be saved to: {output_path}")
        
        # Step 6: Create transcription adapter
        logger.info(f"Creating {selected_engine} adapter...")
        start_time = time.time()
        adapter = factory.create_engine(selected_engine, transcription_config)
        adapter_creation_time = time.time() - start_time
        logging_config.log_operation_timing("Adapter creation", adapter_creation_time)
        
        # Step 7: Run transcription with progress indication
        logger.info(f"Starting transcription of {source}...")
        click.echo(f"Transcribing {source} using {selected_engine}...")
        
        with logging_config.progress_context("Transcribing audio") as progress:
            start_time = time.time()
            raw_transcript = adapter.transcribe(source, language=language)
            transcription_time = time.time() - start_time
            
            progress.update(message="Processing transcript")
            transcript = normalize_transcript_v1(raw_transcript, adapter)
            
            logging_config.log_operation_timing("Transcription", transcription_time)
        
        # Step 8: Save transcript
        with logging_config.progress_context("Saving transcript") as progress:
            logger.info(f"Saving transcript to {output_path}...")
            start_time = time.time()
            strategy = LocalFilePersistence()
            strategy.persist(transcript, str(output_path))
            save_time = time.time() - start_time
            
            logging_config.log_operation_timing("File save", save_time)
            progress.update(message="Transcript saved")
        
        # Step 9: Success message
        click.echo("Transcription completed successfully!")
        click.echo(f"Transcript saved to: {output_path}")
        
        # Show cost information if available
        if hasattr(adapter, 'estimate_cost'):
            try:
                cost_info = adapter.estimate_cost(source)
                if cost_info and cost_info.get('estimated_cost', 0) > 0:
                    click.echo(f"Estimated cost: ${cost_info['estimated_cost']:.4f}")
                    logger.debug(f"Cost estimation: {cost_info}")
            except Exception as e:
                logger.debug(f"Could not get cost information: {e}")
        
        # Log final timing summary
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.info(f"Total operation completed in {total_time:.1f}s")
        
    except click.ClickException as e:
        # Handle Click-specific exceptions (like missing required options)
        if "engine" in str(e).lower() or "required" in str(e).lower():
            # This is likely the missing --engine flag
            show_breaking_change_error("engine_required")
        else:
            # Re-raise other Click exceptions normally
            raise
    except FileNotFoundError as e:
        error_msg = ErrorMessages.format_error(
            ErrorCategory.FILE_ACCESS,
            "file_not_found",
            file_path=str(e)
        )
        logger.error(f"File not found: {e}")
        click.echo(error_msg, err=True)
        sys.exit(ExitCodes.FILE_NOT_FOUND)
    except PermissionError as e:
        error_msg = ErrorMessages.format_error(
            ErrorCategory.FILE_ACCESS,
            "permission_denied",
            file_path=str(e)
        )
        logger.error(f"Permission denied: {e}")
        click.echo(error_msg, err=True)
        sys.exit(ExitCodes.PERMISSION_ERROR)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        error_message = str(e)
        
        # Provide specific guidance based on error type using ErrorMessages
        if "engine" in error_message.lower() and ("not available" in error_message.lower() or "requirements" in error_message.lower()):
            error_msg = ErrorMessages.format_error(
                ErrorCategory.ENGINE_INITIALIZATION,
                "engine_not_available",
                engine_name=engine,
                specific_reason=error_message,
                installation_command="pip install openai-whisper",
                available_engines="whisper-local, whisper-api, auto"
            )
            click.echo(error_msg, err=True)
            sys.exit(ExitCodes.ENGINE_NOT_AVAILABLE)
        elif "api" in error_message.lower() or "key" in error_message.lower() or "auth" in error_message.lower():
            error_msg = ErrorMessages.format_error(
                ErrorCategory.API_AUTHENTICATION,
                "missing_api_key",
                service_name="OpenAI Whisper API",
                env_var_name="OPENAI_API_KEY",
                config_section="whisper_api",
                api_key_url="https://platform.openai.com/api-keys"
            )
            click.echo(error_msg, err=True)
            sys.exit(ExitCodes.AUTHENTICATION_ERROR)
        elif "config" in error_message.lower() or "yaml" in error_message.lower():
            error_msg = ErrorMessages.format_error(
                ErrorCategory.CONFIGURATION,
                "invalid_yaml",
                file_path=config or "configuration file",
                line_number="unknown",
                error_details=error_message
            )
            click.echo(error_msg, err=True)
            sys.exit(ExitCodes.INVALID_CONFIGURATION)
        elif "output" in error_message.lower() or "directory" in error_message.lower():
            error_msg = ErrorMessages.format_error(
                ErrorCategory.FILE_ACCESS,
                "directory_creation_failed",
                directory_path=output_dir or transcription_config.output_dir if 'transcription_config' in locals() else "output directory"
            )
            click.echo(error_msg, err=True)
            sys.exit(ExitCodes.PERMISSION_ERROR)
        else:
            # Generic error with migration summary
            click.echo(f"Transcription failed: {error_message}", err=True)
            click.echo("\nIf you're upgrading from v0.6.0, see migration guide below:", err=True)
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
        
        # Log engine selection details
        available_engines = factory.get_available_engines() if hasattr(factory, 'get_available_engines') else ['whisper-local', 'whisper-api']
        logging_config.log_engine_selection(selected_engine, reason, available_engines)
        
        click.echo(f"Auto-selected engine: {selected_engine}")
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
            
            # Use ErrorMessages for consistent error formatting
            formatted_error = ErrorMessages.format_error(
                ErrorCategory.ENGINE_INITIALIZATION,
                "engine_not_available",
                engine_name=engine,
                specific_reason="\n".join(errors),
                installation_command="pip install openai-whisper" if engine == "whisper-local" else "Set API credentials",
                available_engines="whisper-local, whisper-api, auto"
            )
            click.echo(formatted_error, err=True)
            
            # Provide specific guidance based on error type
            if "api" in error_msg.lower() or "key" in error_msg.lower() or "credential" in error_msg.lower():
                show_breaking_change_error("credentials", f"Engine {engine} requires authentication")
            else:
                # Suggest alternatives
                click.echo("\nTry one of these alternatives:")
                click.echo("  • --engine auto (automatically select best available)")
                click.echo("  • --engine whisper-local (if you have Whisper installed)")
                click.echo("  • Check configuration and credentials")
                sys.exit(ExitCodes.ENGINE_NOT_AVAILABLE)
        
        logger.info(f"Engine {engine} validated successfully")
        
        # Log successful engine selection
        available_engines = factory.get_available_engines() if hasattr(factory, 'get_available_engines') else ['whisper-local', 'whisper-api']
        logging_config.log_engine_selection(engine, "User specified", available_engines)
        
        return engine_type