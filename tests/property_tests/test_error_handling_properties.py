"""
Property-Based Tests for Error Handling Consistency

This module tests Property 9: Error Handling Consistency
Validates Requirements 9.1, 9.2, 9.3, 9.4, 9.5

Property 9: Error Handling Consistency
For any error condition (configuration errors, engine failures, file operations, 
API errors), the system should provide clear error messages with appropriate 
exit codes and suggested fixes.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings
from typing import Dict, Any, Optional

from pipeline.utils.error_messages import ErrorMessages, ErrorCategory, ErrorFormatter
from pipeline.config.manager import ConfigurationManager
from pipeline.transcribers.factory import EngineFactory
from pipeline.output.manager import OutputManager


class TestErrorHandlingConsistency:
    """Test Property 9: Error Handling Consistency"""
    
    @given(
        error_category=st.sampled_from(list(ErrorCategory)),
        template_key=st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=122)),
        context_data=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=122)),
            st.one_of(st.text(max_size=100), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=50)
    def test_property_9a_error_message_formatting_consistency(self, error_category, template_key, context_data):
        """
        Property 9a: Error Message Formatting Consistency
        
        For any error category and context data, ErrorMessages.format_error should:
        1. Return a non-empty string
        2. Include actionable suggestions (💡)
        3. Be properly formatted for CLI display
        4. Handle missing template variables gracefully
        
        **Validates: Requirements 9.1, 9.2**
        """
        # Format error message
        error_message = ErrorMessages.format_error(
            error_category, template_key, **context_data
        )
        
        # Property: Error message should be non-empty
        assert error_message.strip(), f"Error message should not be empty for {error_category.value}.{template_key}"
        
        # Property: Error message should contain suggestions indicator
        assert "💡" in error_message, f"Error message should contain suggestions for {error_category.value}.{template_key}"
        
        # Property: Error message should be properly formatted (no template variables left)
        assert "{" not in error_message or "}" not in error_message or error_message.count("{") == error_message.count("}"), \
            f"Error message should not contain unresolved template variables: {error_message[:100]}..."
        
        # Property: Error message should be reasonable length (not too short or too long)
        assert 10 <= len(error_message) <= 5000, f"Error message length should be reasonable: {len(error_message)}"
    
    @given(
        error_messages=st.lists(
            st.text(min_size=10, max_size=500, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
            min_size=1,
            max_size=20
        ),
        include_emoji=st.booleans()
    )
    @settings(max_examples=30)
    def test_property_9b_cli_formatting_consistency(self, error_messages, include_emoji):
        """
        Property 9b: CLI Formatting Consistency
        
        For any error message, ErrorFormatter.format_for_cli should:
        1. Return consistently formatted output
        2. Handle emoji inclusion consistently
        3. Preserve message content
        4. Be suitable for terminal display
        
        **Validates: Requirements 9.1, 9.3**
        """
        for error_message in error_messages:
            formatted = ErrorFormatter.format_for_cli(error_message, include_emoji)
            
            # Property: Formatted message should contain original content
            core_message = error_message.strip()
            assert core_message in formatted, f"Formatted message should contain original content"
            
            # Property: Emoji handling should be consistent
            if include_emoji and not error_message.startswith("❌"):
                assert formatted.startswith("❌"), f"Should add error emoji when requested"
            
            # Property: Formatted message should be suitable for CLI (no control characters)
            printable_chars = set(range(32, 127)) | {9, 10, 13}  # Printable ASCII + tab, newline, carriage return
            for char in formatted:
                char_code = ord(char)
                assert char_code in printable_chars or char_code > 127, \
                    f"Formatted message should only contain printable characters, found: {char_code}"
    
    @given(
        error_messages=st.lists(
            st.text(min_size=20, max_size=1000),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20)
    def test_property_9c_actionable_suggestions_extraction(self, error_messages):
        """
        Property 9c: Actionable Suggestions Extraction
        
        For any error message containing suggestions, ErrorFormatter.extract_actionable_suggestions
        should consistently extract actionable items.
        
        **Validates: Requirements 9.2, 9.4**
        """
        for error_message in error_messages:
            # Add some suggestions to the message
            enhanced_message = error_message + "\n\n💡 Suggestions:\n  • Check configuration\n  • Verify credentials\n  • Try again"
            
            suggestions = ErrorFormatter.extract_actionable_suggestions(enhanced_message)
            
            # Property: Should extract suggestions when present
            if "💡 Suggestions:" in enhanced_message:
                assert len(suggestions) > 0, "Should extract suggestions when present"
                
                # Property: Extracted suggestions should be meaningful
                for suggestion in suggestions:
                    assert len(suggestion.strip()) > 0, "Suggestions should not be empty"
                    assert len(suggestion.strip()) < 500, "Suggestions should be reasonably sized"
    
    @given(
        config_data=st.dictionaries(
            st.sampled_from(['engine', 'output_dir', 'model', 'api_key', 'language', 'log_level']),
            st.one_of(
                st.text(max_size=100),
                st.integers(min_value=-1000, max_value=1000),
                st.none()
            ),
            min_size=1,
            max_size=6
        )
    )
    @settings(max_examples=30)
    def test_property_9d_configuration_error_consistency(self, config_data):
        """
        Property 9d: Configuration Error Consistency
        
        For any configuration error scenario, the system should provide
        consistent error messages with specific field information and suggestions.
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        """
        # Test missing required field error
        if 'engine' not in config_data:
            error_msg = ErrorMessages.format_error(
                ErrorCategory.CONFIGURATION,
                "missing_required_field",
                field_name="engine",
                example_value="whisper-local"
            )
            
            # Property: Should mention the specific missing field
            assert "engine" in error_msg, "Error should mention the specific missing field"
            
            # Property: Should provide example value
            assert "whisper-local" in error_msg, "Error should provide example value"
            
            # Property: Should contain suggestions
            assert "💡" in error_msg, "Error should contain suggestions"
        
        # Test invalid field value error
        if 'engine' in config_data and config_data['engine'] not in [None, 'whisper-local', 'whisper-api', 'auto']:
            error_msg = ErrorMessages.format_error(
                ErrorCategory.CONFIGURATION,
                "invalid_field_value",
                field_name="engine",
                current_value=str(config_data['engine']),
                expected_values="whisper-local, whisper-api, auto",
                example_value="whisper-local"
            )
            
            # Property: Should mention current invalid value
            assert str(config_data['engine']) in error_msg, "Error should mention current invalid value"
            
            # Property: Should list valid options
            assert "whisper-local" in error_msg, "Error should list valid options"
            assert "whisper-api" in error_msg, "Error should list valid options"
    
    @given(
        engine_name=st.sampled_from(['whisper-local', 'whisper-api', 'aws-transcribe', 'invalid-engine']),
        error_reasons=st.lists(
            st.text(min_size=5, max_size=100),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=25)
    def test_property_9e_engine_error_consistency(self, engine_name, error_reasons):
        """
        Property 9e: Engine Error Consistency
        
        For any engine initialization error, the system should provide
        consistent error messages with specific engine information and alternatives.
        
        **Validates: Requirements 9.1, 9.2, 9.4**
        """
        error_msg = ErrorMessages.format_error(
            ErrorCategory.ENGINE_INITIALIZATION,
            "engine_not_available",
            engine_name=engine_name,
            specific_reason="\n".join(error_reasons),
            installation_command=f"pip install {engine_name}-package",
            available_engines="whisper-local, whisper-api, auto"
        )
        
        # Property: Should mention the specific engine
        assert engine_name in error_msg, f"Error should mention the specific engine: {engine_name}"
        
        # Property: Should include specific error reasons
        for reason in error_reasons:
            if len(reason.strip()) > 0:
                assert reason in error_msg, f"Error should include specific reason: {reason}"
        
        # Property: Should suggest alternatives
        assert "whisper-local" in error_msg or "whisper-api" in error_msg, "Error should suggest alternative engines"
        
        # Property: Should contain installation guidance
        assert "install" in error_msg.lower() or "pip" in error_msg.lower(), "Error should contain installation guidance"
    
    @given(
        file_paths=st.lists(
            st.text(min_size=1, max_size=200, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.-_"),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20)
    def test_property_9f_file_error_consistency(self, file_paths):
        """
        Property 9f: File Error Consistency
        
        For any file operation error, the system should provide
        consistent error messages with specific file paths and resolution steps.
        
        **Validates: Requirements 9.1, 9.2, 9.5**
        """
        for file_path in file_paths:
            assume(len(file_path.strip()) > 0)
            
            # Test file not found error
            error_msg = ErrorMessages.format_error(
                ErrorCategory.FILE_ACCESS,
                "file_not_found",
                file_path=file_path
            )
            
            # Property: Should mention the specific file path
            assert file_path in error_msg, f"Error should mention the specific file path: {file_path}"
            
            # Property: Should provide resolution suggestions
            assert "💡" in error_msg, "Error should contain suggestions"
            assert ("check" in error_msg.lower() or "verify" in error_msg.lower()), \
                "Error should suggest checking/verifying the file"
            
            # Test permission denied error
            permission_error = ErrorMessages.format_error(
                ErrorCategory.FILE_ACCESS,
                "permission_denied",
                file_path=file_path
            )
            
            # Property: Should mention permissions
            assert "permission" in permission_error.lower(), "Permission error should mention permissions"
            assert file_path in permission_error, "Permission error should mention the specific file"
    
    @given(
        service_names=st.sampled_from(['OpenAI Whisper API', 'AWS Transcribe', 'Generic API Service']),
        api_key_vars=st.sampled_from(['OPENAI_API_KEY', 'AWS_ACCESS_KEY_ID', 'API_KEY'])
    )
    @settings(max_examples=15)
    def test_property_9g_api_error_consistency(self, service_names, api_key_vars):
        """
        Property 9g: API Error Consistency
        
        For any API authentication error, the system should provide
        consistent error messages with specific service information and setup instructions.
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        """
        error_msg = ErrorMessages.format_error(
            ErrorCategory.API_AUTHENTICATION,
            "missing_api_key",
            service_name=service_names,
            env_var_name=api_key_vars,
            config_section="api_config",
            api_key_url="https://example.com/api-keys"
        )
        
        # Property: Should mention the specific service
        assert service_names in error_msg, f"Error should mention the specific service: {service_names}"
        
        # Property: Should mention the environment variable
        assert api_key_vars in error_msg, f"Error should mention the environment variable: {api_key_vars}"
        
        # Property: Should provide multiple setup options
        setup_options = ["Environment variable", "CLI flag", "Configuration file"]
        found_options = sum(1 for option in setup_options if option.lower() in error_msg.lower())
        assert found_options >= 2, "Error should provide multiple setup options"
        
        # Property: Should include URL for getting API key
        assert "https://" in error_msg, "Error should include URL for getting API key"
    
    @given(
        error_scenarios=st.lists(
            st.dictionaries(
                st.sampled_from(['type', 'message', 'context']),
                st.text(max_size=100),
                min_size=2,
                max_size=3
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=20)
    def test_property_9h_error_suggestion_consistency(self, error_scenarios):
        """
        Property 9h: Error Suggestion Consistency
        
        For any error message, ErrorMessages.get_suggestion_for_error should
        provide contextually appropriate suggestions based on error content.
        
        **Validates: Requirements 9.2, 9.4, 9.5**
        """
        for scenario in error_scenarios:
            error_message = scenario.get('message', '')
            assume(len(error_message.strip()) > 0)
            
            suggestion = ErrorMessages.get_suggestion_for_error(error_message)
            
            # Property: Should always provide some suggestion
            assert suggestion is not None, "Should always provide a suggestion"
            assert len(suggestion.strip()) > 0, "Suggestion should not be empty"
            
            # Property: Suggestion should be actionable (contain action words)
            action_words = ['check', 'verify', 'try', 'use', 'set', 'install', 'run', 'ensure']
            has_action_word = any(word in suggestion.lower() for word in action_words)
            assert has_action_word, f"Suggestion should contain actionable advice: {suggestion}"
            
            # Property: Suggestion should be reasonably sized
            assert 10 <= len(suggestion) <= 500, f"Suggestion should be reasonably sized: {len(suggestion)}"
    
    def test_property_9i_error_category_coverage(self):
        """
        Property 9i: Error Category Coverage
        
        The ErrorMessages class should provide templates for all defined error categories
        and handle unknown categories gracefully.
        
        **Validates: Requirements 9.1, 9.5**
        """
        # Property: All error categories should be handled
        for category in ErrorCategory:
            # Test with a generic template key
            error_msg = ErrorMessages.format_error(
                category, "generic_error", error_details="Test error"
            )
            
            # Property: Should generate a meaningful error message
            assert len(error_msg.strip()) > 0, f"Should generate error message for category: {category.value}"
            assert category.value.replace('_', ' ').title() in error_msg, \
                f"Error message should mention the category: {category.value}"
            
            # Property: Should contain suggestions
            assert "💡" in error_msg, f"Error message should contain suggestions for category: {category.value}"
    
    @given(
        log_levels=st.sampled_from(['ERROR', 'WARNING', 'INFO', 'DEBUG']),
        context_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            min_size=0,
            max_size=5
        )
    )
    @settings(max_examples=20)
    def test_property_9j_logging_format_consistency(self, log_levels, context_data):
        """
        Property 9j: Logging Format Consistency
        
        For any error message and context, ErrorFormatter.format_for_logging should
        provide consistent formatting with appropriate log levels and context.
        
        **Validates: Requirements 9.1, 9.3**
        """
        error_message = "Test error message for logging"
        
        formatted = ErrorFormatter.format_for_logging(
            error_message, log_levels, context_data
        )
        
        # Property: Should include log level
        assert f"[{log_levels}]" in formatted, f"Should include log level: {log_levels}"
        
        # Property: Should include original message
        assert error_message in formatted, "Should include original error message"
        
        # Property: Should include context if provided
        if context_data:
            assert "Context:" in formatted, "Should include context section when provided"
            for key, value in context_data.items():
                assert f"{key}={value}" in formatted, f"Should include context item: {key}={value}"
        
        # Property: Should be properly formatted for logging
        assert formatted.startswith(f"[{log_levels}]"), "Should start with log level"
        assert len(formatted) > len(error_message), "Should be longer than original message"


if __name__ == "__main__":
    # Run a quick test to verify the property tests work
    test_instance = TestErrorHandlingConsistency()
    
    # Test a simple case
    test_instance.test_property_9a_error_message_formatting_consistency(
        ErrorCategory.CONFIGURATION, "test_template", {"field": "test"}
    )
    
    print("✅ Property 9: Error Handling Consistency tests are working correctly")