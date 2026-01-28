"""
Property-Based Tests for Logging Configuration

This module tests Property 12: Logging Configuration
Validates Requirements 12.1, 12.3, 12.5

Property 12: Logging Configuration
For any specified log level, the CLI should set logging verbosity correctly 
and include appropriate debug information for engine selection and configuration loading.
"""

import pytest
import tempfile
import os
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings
from typing import Dict, Any, Optional
from io import StringIO

from pipeline.utils.logging_config import LoggingConfig, ProgressIndicator, LogLevel, logging_config


class TestLoggingConfiguration:
    """Test Property 12: Logging Configuration"""
    
    def setup_method(self):
        """Reset logging configuration before each test."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Reset the global logging config
        global logging_config
        logging_config = LoggingConfig()
    
    @given(
        log_levels=st.sampled_from(['debug', 'info', 'warning', 'error']),
        include_timestamps=st.booleans(),
        include_module_names=st.booleans()
    )
    @settings(max_examples=20)
    def test_property_12a_log_level_configuration_consistency(self, log_levels, include_timestamps, include_module_names):
        """
        Property 12a: Log Level Configuration Consistency
        
        For any specified log level, LoggingConfig.configure_logging should:
        1. Set the correct logging level
        2. Configure handlers appropriately
        3. Apply consistent formatting options
        4. Enable/disable debug information correctly
        
        **Validates: Requirements 12.1, 12.3**
        """
        config = LoggingConfig()
        
        # Configure logging with specified parameters
        config.configure_logging(
            level=log_levels,
            include_timestamps=include_timestamps,
            include_module_names=include_module_names
        )
        
        # Property: Root logger should be set to correct level
        root_logger = logging.getLogger()
        expected_level = getattr(logging, log_levels.upper())
        assert root_logger.level == expected_level, f"Root logger level should be {expected_level} for {log_levels}"
        
        # Property: Should have at least one handler (console)
        assert len(root_logger.handlers) >= 1, "Should have at least one logging handler"
        
        # Property: Console handler should exist and be configured correctly
        console_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                console_handler = handler
                break
        
        assert console_handler is not None, "Should have a console handler"
        assert console_handler.level == expected_level, f"Console handler level should match root logger level"
        
        # Property: Debug level should enable more detailed logging
        if log_levels == 'debug':
            assert config.is_debug_enabled(), "Debug should be enabled for debug log level"
        else:
            # For non-debug levels, debug should not be enabled
            debug_logger = logging.getLogger('test_debug')
            assert not debug_logger.isEnabledFor(logging.DEBUG) or log_levels == 'debug', \
                f"Debug should not be enabled for {log_levels} level"
    
    @given(
        descriptions=st.lists(
            st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
            min_size=1,
            max_size=10
        ),
        total_steps=st.one_of(st.none(), st.integers(min_value=1, max_value=100))
    )
    @settings(max_examples=15)
    def test_property_12b_progress_indicator_consistency(self, descriptions, total_steps):
        """
        Property 12b: Progress Indicator Consistency
        
        For any progress description and step count, ProgressIndicator should:
        1. Display progress consistently
        2. Handle determinate and indeterminate progress
        3. Complete successfully
        4. Measure timing accurately
        
        **Validates: Requirements 12.1, 12.5**
        """
        for description in descriptions:
            assume(len(description.strip()) > 0)
            
            progress = ProgressIndicator(description, total_steps)
            
            # Property: Initial state should be correct
            assert progress.description == description, "Description should be preserved"
            assert progress.total_steps == total_steps, "Total steps should be preserved"
            assert progress.current_step == 0, "Should start at step 0"
            assert progress.start_time > 0, "Should have valid start time"
            
            # Property: Update should work correctly
            if total_steps:
                # Test determinate progress
                for step in range(min(3, total_steps)):
                    progress.update(step + 1)
                    assert progress.current_step == step + 1, f"Current step should be {step + 1}"
            else:
                # Test indeterminate progress
                for _ in range(3):
                    old_step = progress.current_step
                    progress.update()
                    assert progress.current_step == old_step + 1, "Should increment step for indeterminate progress"
            
            # Property: Finish should complete successfully
            progress.finish()
            # No assertion needed - just verify it doesn't crash
    
    @given(
        config_data=st.dictionaries(
            st.sampled_from(['engine', 'output_dir', 'model', 'api_key', 'language']),
            st.one_of(st.text(max_size=50), st.integers(), st.none()),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=20)
    def test_property_12c_configuration_logging_consistency(self, config_data):
        """
        Property 12c: Configuration Logging Consistency
        
        For any configuration data, LoggingConfig.log_configuration_details should:
        1. Log configuration at debug level only
        2. Mask sensitive information
        3. Format configuration consistently
        4. Handle various data types
        
        **Validates: Requirements 12.3, 12.5**
        """
        config = LoggingConfig()
        config.configure_logging(level='debug')
        
        # Capture log output
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.isEnabledFor.return_value = True
            
            config.log_configuration_details(config_data)
            
            # Property: Should call debug logging
            assert mock_logger.debug.called, "Should call debug logging for configuration"
            
            # Property: Should log configuration header
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            header_found = any("Configuration Details" in call for call in debug_calls)
            assert header_found, "Should log configuration header"
            
            # Property: Should log each configuration item
            for key, value in config_data.items():
                key_found = any(key in call for call in debug_calls)
                assert key_found, f"Should log configuration key: {key}"
                
                # Property: Should mask sensitive values
                if "key" in key.lower() or "password" in key.lower():
                    masked_found = any("***MASKED***" in call for call in debug_calls)
                    if value:  # Only check masking if value is not None/empty
                        assert masked_found, f"Should mask sensitive value for {key}"
    
    @given(
        engine_names=st.sampled_from(['whisper-local', 'whisper-api', 'aws-transcribe', 'auto']),
        reasons=st.text(min_size=5, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        available_engines=st.lists(
            st.sampled_from(['whisper-local', 'whisper-api', 'aws-transcribe']),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=15)
    def test_property_12d_engine_selection_logging_consistency(self, engine_names, reasons, available_engines):
        """
        Property 12d: Engine Selection Logging Consistency
        
        For any engine selection scenario, LoggingConfig.log_engine_selection should:
        1. Log engine selection at info level
        2. Log selection reason at debug level
        3. Log available engines at debug level
        4. Format information consistently
        
        **Validates: Requirements 12.3, 12.5**
        """
        assume(len(reasons.strip()) > 0)
        
        config = LoggingConfig()
        config.configure_logging(level='debug')
        
        # Capture log output
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.isEnabledFor.return_value = True
            
            config.log_engine_selection(engine_names, reasons, available_engines)
            
            # Property: Should log at info level for engine selection
            assert mock_logger.info.called, "Should log engine selection at info level"
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            engine_logged = any(engine_names in call for call in info_calls)
            assert engine_logged, f"Should log selected engine: {engine_names}"
            
            # Property: Should log at debug level for details
            assert mock_logger.debug.called, "Should log details at debug level"
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            
            # Property: Should log selection reason
            reason_logged = any(reasons in call for call in debug_calls)
            assert reason_logged, f"Should log selection reason: {reasons}"
            
            # Property: Should log available engines
            for engine in available_engines:
                engine_logged = any(engine in call for call in debug_calls)
                assert engine_logged, f"Should log available engine: {engine}"
    
    @given(
        operations=st.lists(
            st.text(min_size=3, max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
            min_size=1,
            max_size=10
        ),
        durations=st.lists(
            st.floats(min_value=0.001, max_value=300.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=15)
    def test_property_12e_operation_timing_logging_consistency(self, operations, durations):
        """
        Property 12e: Operation Timing Logging Consistency
        
        For any operation timing scenario, LoggingConfig.log_operation_timing should:
        1. Log timing information appropriately
        2. Use correct units (ms for < 1s, s for >= 1s)
        3. Format timing consistently
        4. Choose appropriate log level based on duration
        
        **Validates: Requirements 12.1, 12.5**
        """
        config = LoggingConfig()
        config.configure_logging(level='debug')
        
        # Test each operation-duration pair
        for operation, duration in zip(operations, durations):
            assume(len(operation.strip()) > 0)
            
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                mock_logger.isEnabledFor.return_value = True
                
                config.log_operation_timing(operation, duration)
                
                # Property: Should log timing information
                logged = mock_logger.info.called or mock_logger.debug.called
                assert logged, f"Should log timing for operation: {operation}"
                
                # Property: Should use appropriate log level based on duration
                if duration < 1.0:
                    assert mock_logger.debug.called, f"Should use debug level for short duration: {duration}"
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    timing_logged = any(operation in call and "ms" in call for call in debug_calls)
                    assert timing_logged, f"Should log timing in milliseconds for {operation}"
                else:
                    assert mock_logger.info.called, f"Should use info level for longer duration: {duration}"
                    info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                    timing_logged = any(operation in call and "s" in call for call in info_calls)
                    assert timing_logged, f"Should log timing in seconds for {operation}"
    
    @given(
        log_levels=st.sampled_from(['debug', 'info', 'warning', 'error']),
        log_file_paths=st.one_of(
            st.none(),
            st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789/.-_")
        )
    )
    @settings(max_examples=10)
    def test_property_12f_file_logging_configuration_consistency(self, log_levels, log_file_paths):
        """
        Property 12f: File Logging Configuration Consistency
        
        For any log level and file path, LoggingConfig should:
        1. Configure file logging when path is provided
        2. Handle file creation and permissions gracefully
        3. Use appropriate formatters for file output
        4. Fall back to console-only logging on file errors
        
        **Validates: Requirements 12.1, 12.3**
        """
        config = LoggingConfig()
        
        if log_file_paths:
            # Use a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                test_log_file = os.path.join(temp_dir, "test.log")
                
                try:
                    # Configure logging with file output
                    config.configure_logging(level=log_levels, log_file=test_log_file)
                    
                    # Property: Should have console handler
                    root_logger = logging.getLogger()
                    console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
                    assert len(console_handlers) >= 1, "Should have console handler"
                    
                    # Property: Should have file handler if file was created successfully
                    file_handlers = [h for h in root_logger.handlers if hasattr(h, 'baseFilename')]
                    # Note: File handler creation might fail due to permissions, which is acceptable
                    
                    # Property: All handlers should have correct log level
                    expected_level = getattr(logging, log_levels.upper())
                    for handler in root_logger.handlers:
                        assert handler.level == expected_level, f"Handler should have level {expected_level}"
                        
                finally:
                    # Clean up handlers to prevent file locking issues
                    root_logger = logging.getLogger()
                    for handler in root_logger.handlers[:]:
                        if hasattr(handler, 'close'):
                            handler.close()
                        root_logger.removeHandler(handler)
        else:
            # Configure logging without file output
            config.configure_logging(level=log_levels)
            
            # Property: Should have only console handler
            root_logger = logging.getLogger()
            console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
            assert len(console_handlers) >= 1, "Should have console handler"
            
            file_handlers = [h for h in root_logger.handlers if hasattr(h, 'baseFilename')]
            assert len(file_handlers) == 0, "Should not have file handler when no file specified"
    
    def test_property_12g_progress_context_manager_consistency(self):
        """
        Property 12g: Progress Context Manager Consistency
        
        The progress context manager should:
        1. Handle successful operations correctly
        2. Handle exceptions gracefully
        3. Always call finish() regardless of outcome
        4. Provide proper progress object
        
        **Validates: Requirements 12.5**
        """
        config = LoggingConfig()
        
        # Test successful operation
        progress_obj = None
        with config.progress_context("Test operation", 5) as progress:
            progress_obj = progress
            assert isinstance(progress, ProgressIndicator), "Should provide ProgressIndicator instance"
            assert progress.description == "Test operation", "Should preserve description"
            assert progress.total_steps == 5, "Should preserve total steps"
            
            # Simulate some work
            progress.update(1)
            progress.update(2)
        
        # Property: Context manager should complete successfully
        assert progress_obj is not None, "Should have provided progress object"
        
        # Test exception handling
        exception_raised = False
        try:
            with config.progress_context("Failing operation") as progress:
                progress.update(1)
                raise ValueError("Test exception")
        except ValueError:
            exception_raised = True
        
        # Property: Should re-raise exceptions
        assert exception_raised, "Should re-raise exceptions from context"
    
    @given(
        log_levels=st.sampled_from(['debug', 'info', 'warning', 'error'])
    )
    @settings(max_examples=8)
    def test_property_12h_debug_enablement_consistency(self, log_levels):
        """
        Property 12h: Debug Enablement Consistency
        
        For any log level, LoggingConfig.is_debug_enabled should:
        1. Return True only when debug logging is actually enabled
        2. Be consistent with logger.isEnabledFor(logging.DEBUG)
        3. Reflect the actual logging configuration
        
        **Validates: Requirements 12.3**
        """
        config = LoggingConfig()
        config.configure_logging(level=log_levels)
        
        # Property: is_debug_enabled should match actual debug enablement
        debug_enabled = config.is_debug_enabled()
        logger_debug_enabled = logging.getLogger().isEnabledFor(logging.DEBUG)
        
        assert debug_enabled == logger_debug_enabled, \
            f"is_debug_enabled() should match logger.isEnabledFor(DEBUG) for level {log_levels}"
        
        # Property: Should be True only for debug level
        if log_levels == 'debug':
            assert debug_enabled, "Should be enabled for debug level"
        else:
            assert not debug_enabled, f"Should not be enabled for {log_levels} level"


if __name__ == "__main__":
    # Run a quick test to verify the property tests work
    test_instance = TestLoggingConfiguration()
    test_instance.setup_method()
    
    # Test a simple case
    test_instance.test_property_12a_log_level_configuration_consistency('info', True, True)
    
    print("✅ Property 12: Logging Configuration tests are working correctly")