"""
Logging Configuration and Progress Reporting

This module provides configurable logging levels and progress indicators
for the content pipeline system.

Enhanced in v0.6.5 to support:
- Configurable logging levels (debug, info, warning, error)
- Progress indicators for long operations
- Debug logging for engine selection and configuration
- Optional log file output with rotation
- Integration with error message templates
"""

import logging
import logging.handlers
import sys
import time
from typing import Optional, Dict, Any, TextIO
from pathlib import Path
from contextlib import contextmanager
from enum import Enum


class LogLevel(Enum):
    """Supported logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ProgressIndicator:
    """
    Simple progress indicator for long-running operations.
    
    Provides visual feedback during transcription and other operations.
    """
    
    def __init__(self, description: str, total_steps: Optional[int] = None):
        self.description = description
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self._last_update = 0
        
    def update(self, step: Optional[int] = None, message: Optional[str] = None):
        """Update progress indicator."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        # Only update display every 0.5 seconds to avoid spam
        current_time = time.time()
        if current_time - self._last_update < 0.5:
            return
            
        self._last_update = current_time
        elapsed = current_time - self.start_time
        
        if self.total_steps:
            percentage = (self.current_step / self.total_steps) * 100
            progress_bar = self._create_progress_bar(percentage)
            status = f"{progress_bar} {percentage:.1f}% ({self.current_step}/{self.total_steps})"
        else:
            # Indeterminate progress
            spinner = self._get_spinner_char()
            status = f"{spinner} Step {self.current_step}"
        
        display_message = message or self.description
        elapsed_str = f"{elapsed:.1f}s"
        
        # Clear line and print progress
        print(f"\r{display_message} {status} [{elapsed_str}]", end="", flush=True)
    
    def finish(self, message: Optional[str] = None):
        """Complete the progress indicator."""
        elapsed = time.time() - self.start_time
        final_message = message or f"{self.description} completed"
        print(f"\r{final_message} [OK] [{elapsed:.1f}s]")
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(width * percentage / 100)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}]"
    
    def _get_spinner_char(self) -> str:
        """Get rotating spinner character."""
        chars = "|/-\\"
        return chars[self.current_step % len(chars)]


class LoggingConfig:
    """
    Centralized logging configuration for the content pipeline.
    
    Provides configurable logging levels, optional file output,
    and integration with progress reporting.
    """
    
    def __init__(self):
        self._configured = False
        self._log_file_handler: Optional[logging.Handler] = None
        self._console_handler: Optional[logging.Handler] = None
        
    def configure_logging(
        self,
        level: str = "info",
        log_file: Optional[str] = None,
        include_timestamps: bool = True,
        include_module_names: bool = True,
        max_log_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Configure logging for the application.
        
        Args:
            level: Logging level (debug, info, warning, error)
            log_file: Optional log file path
            include_timestamps: Whether to include timestamps in log messages
            include_module_names: Whether to include module names
            max_log_file_size: Maximum log file size before rotation
            backup_count: Number of backup log files to keep
        """
        if self._configured:
            return
            
        # Convert string level to logging constant
        log_level = self._get_log_level(level)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        console_formatter = self._create_console_formatter(
            include_timestamps, include_module_names, level == "debug"
        )
        file_formatter = self._create_file_formatter()
        
        # Configure console handler
        self._console_handler = logging.StreamHandler(sys.stderr)
        self._console_handler.setLevel(log_level)
        self._console_handler.setFormatter(console_formatter)
        root_logger.addHandler(self._console_handler)
        
        # Configure file handler if requested
        if log_file:
            self._configure_file_logging(
                log_file, file_formatter, log_level, 
                max_log_file_size, backup_count
            )
        
        self._configured = True
        
        # Log configuration details at debug level
        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured: level={level}, file={log_file}")
        
    def _get_log_level(self, level_str: str) -> int:
        """Convert string log level to logging constant."""
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        return level_map.get(level_str.lower(), logging.INFO)
    
    def _create_console_formatter(
        self, 
        include_timestamps: bool, 
        include_module_names: bool,
        debug_mode: bool
    ) -> logging.Formatter:
        """Create formatter for console output."""
        parts = []
        
        if include_timestamps:
            parts.append("%(asctime)s")
            
        if debug_mode and include_module_names:
            parts.append("%(name)s")
            
        parts.extend(["%(levelname)s", "%(message)s"])
        
        format_str = " - ".join(parts)
        
        formatter = logging.Formatter(
            format_str,
            datefmt="%H:%M:%S" if not debug_mode else "%Y-%m-%d %H:%M:%S"
        )
        
        return formatter
    
    def _create_file_formatter(self) -> logging.Formatter:
        """Create formatter for file output."""
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def _configure_file_logging(
        self,
        log_file: str,
        formatter: logging.Formatter,
        log_level: int,
        max_size: int,
        backup_count: int
    ) -> None:
        """Configure file logging with rotation."""
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            self._log_file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            self._log_file_handler.setLevel(log_level)
            self._log_file_handler.setFormatter(formatter)
            
            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(self._log_file_handler)
            
        except Exception as e:
            # Log file setup failed, continue with console only
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to setup log file {log_file}: {e}")
    
    @contextmanager
    def progress_context(self, description: str, total_steps: Optional[int] = None):
        """
        Context manager for progress indication.
        
        Usage:
            with logging_config.progress_context("Processing files", 10) as progress:
                for i in range(10):
                    # Do work
                    progress.update(message=f"Processing file {i+1}")
        """
        progress = ProgressIndicator(description, total_steps)
        try:
            yield progress
        except Exception as e:
            progress.finish(f"{description} failed: {e}")
            raise
        else:
            progress.finish()
    
    def log_configuration_details(self, config: Dict[str, Any]) -> None:
        """Log configuration details at debug level."""
        logger = logging.getLogger(__name__)
        
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        logger.debug("=== Configuration Details ===")
        for key, value in config.items():
            # Mask sensitive values
            if "key" in key.lower() or "password" in key.lower() or "secret" in key.lower():
                display_value = "***MASKED***" if value else None
            else:
                display_value = value
            logger.debug(f"  {key}: {display_value}")
        logger.debug("=== End Configuration ===")
    
    def log_engine_selection(self, engine: str, reason: str, available_engines: list) -> None:
        """Log engine selection details at debug level."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"Selected engine: {engine}")
        logger.debug(f"Selection reason: {reason}")
        logger.debug(f"Available engines: {', '.join(available_engines)}")
    
    def log_operation_timing(self, operation: str, duration: float) -> None:
        """Log operation timing information."""
        logger = logging.getLogger(__name__)
        
        if duration < 1.0:
            logger.debug(f"{operation} completed in {duration*1000:.0f}ms")
        else:
            logger.info(f"{operation} completed in {duration:.1f}s")
    
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return logging.getLogger().isEnabledFor(logging.DEBUG)


# Global logging configuration instance
logging_config = LoggingConfig()


def configure_logging(level: str = "info", log_file: Optional[str] = None) -> None:
    """
    Convenience function to configure logging.
    
    Args:
        level: Logging level (debug, info, warning, error)
        log_file: Optional log file path
    """
    logging_config.configure_logging(level=level, log_file=log_file)


def get_progress_context(description: str, total_steps: Optional[int] = None):
    """
    Convenience function to get progress context.
    
    Args:
        description: Description of the operation
        total_steps: Total number of steps (None for indeterminate)
        
    Returns:
        Progress context manager
    """
    return logging_config.progress_context(description, total_steps)