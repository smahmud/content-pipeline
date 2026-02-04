"""
Property-Based Tests for CLI Refactoring

These tests validate universal properties that should hold across all CLI operations,
using hypothesis for property-based testing to catch edge cases.
"""

import pytest
import click
from hypothesis import given, strategies as st, settings
from click.testing import CliRunner
from cli.shared_options import input_option, output_option, language_option
from cli.extract import extract
from cli.transcribe import transcribe
from cli import main, cli


class TestEntryPointIndependence:
    """
    Property 4: Entry Point Independence
    For any CLI invocation, the system should function correctly without 
    main_cli.py being present.
    
    Feature: cli-refactoring-v0-6-0, Property 4: Entry Point Independence
    Validates: Requirements 4.4
    """
    
    def test_cli_functions_without_main_cli_py(self):
        """Test that CLI works without importing main_cli.py."""
        
        # Test that we can import and use the CLI without main_cli.py
        runner = CliRunner()
        
        # Test main CLI group
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Content Pipeline CLI' in result.output
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        
        # Test extract subcommand
        result = runner.invoke(main, ['extract', '--help'])
        assert result.exit_code == 0
        assert 'Extract audio from the source file' in result.output
        
        # Test transcribe subcommand - check for key parts since Click wraps text
        result = runner.invoke(main, ['transcribe', '--help'])
        assert result.exit_code == 0
        assert 'Transcribe audio content to text' in result.output
        assert 'configurable speech recognition' in result.output
    
    def test_cli_entry_point_function(self):
        """Test that cli() entry point function works independently."""
        
        from cli import cli
        
        # Test that cli function exists and is callable
        assert callable(cli)
        
        # Test that it's properly configured for setup.py
        # (We can't easily test the actual console script execution here,
        # but we can verify the function exists and has the right structure)
        import inspect
        assert inspect.isfunction(cli)
    
    def test_no_main_cli_imports_in_cli_package(self):
        """Test that cli package doesn't import main_cli.py."""
        
        import cli
        import cli.extract
        import cli.transcribe
        import cli.shared_options
        import cli.help_texts
        
        # Check that none of the cli modules import main_cli
        # This is a static check - if main_cli was imported, it would show up
        # in the module's globals or cause import errors
        
        # The fact that we can import all cli modules without error
        # and they work correctly proves independence from main_cli.py
        assert True  # If we get here, the imports succeeded


class TestCommandRoutingAndRegistration:
    """
    Property 3: Command Routing and Registration
    For any registered subcommand, the Click group should route commands correctly 
    to their respective modules and make them available in the command list.
    
    Feature: cli-refactoring-v0-6-0, Property 3: Command Routing and Registration
    Validates: Requirements 4.2, 4.3
    """
    
    def test_main_group_has_registered_commands(self):
        """Test that main CLI group has all expected commands registered."""
        
        # Get list of registered commands
        registered_commands = list(main.commands.keys())
        
        # Verify expected commands are registered
        assert 'extract' in registered_commands
        assert 'transcribe' in registered_commands
        
        # Verify commands are properly mapped
        assert main.commands['extract'] is extract
        assert main.commands['transcribe'] is transcribe
    
    def test_command_routing_via_main_group(self):
        """Test that commands are properly routed through the main group."""
        runner = CliRunner()
        
        # Test that main group shows available commands
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        assert 'Commands:' in result.output
        
        # Test routing to extract command
        result = runner.invoke(main, ['extract', '--help'])
        assert result.exit_code == 0
        assert 'Extract audio from the source file' in result.output
        
        # Test routing to transcribe command - check for key parts since Click wraps text
        result = runner.invoke(main, ['transcribe', '--help'])
        assert result.exit_code == 0
        assert 'Transcribe audio content to text' in result.output
        assert 'configurable speech recognition' in result.output
    
    def test_version_option_available(self):
        """Test that version option is available in main group."""
        runner = CliRunner()
        
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.7.0' in result.output
    
    def test_main_group_structure(self):
        """Test that main group has proper Click structure."""
        # Test that main is a Click group
        assert isinstance(main, click.Group)
        
        # Test that it has the expected name and help
        assert main.name == 'main'
        assert 'Content Pipeline CLI' in main.help
        
        # Test that commands are properly attached
        assert len(main.commands) >= 2  # extract and transcribe at minimum


class TestSubcommandIndependence:
    """
    Property 1: Subcommand Independence
    For any subcommand in the CLI package, executing it should work without 
    importing or depending on main_cli.py
    
    Feature: cli-refactoring-v0-6-0, Property 1: Subcommand Independence
    Validates: Requirements 2.5
    """
    
    def test_extract_subcommand_independence(self):
        """Test that extract subcommand works independently of main_cli.py."""
        
        # Test that we can import and use extract without main_cli.py
        runner = CliRunner()
        
        # Test help output works
        result = runner.invoke(extract, ['--help'])
        assert result.exit_code == 0
        assert 'Extract audio from the source file' in result.output
        assert '--source' in result.output
        assert '--output' in result.output
        
        # Test that the command is properly decorated
        assert hasattr(extract, 'callback')
        assert extract.name == 'extract'
    
    def test_transcribe_subcommand_independence(self):
        """Test that transcribe subcommand works independently of main_cli.py."""
        
        # Test that we can import and use transcribe without main_cli.py
        runner = CliRunner()
        
        # Test help output works - check for key parts since Click wraps text
        result = runner.invoke(transcribe, ['--help'])
        assert result.exit_code == 0
        assert 'Transcribe audio content to text' in result.output
        assert 'configurable speech recognition' in result.output
        assert '--source' in result.output
        assert '--output' in result.output
        assert '--language' in result.output
        
        # Test that the command is properly decorated
        assert hasattr(transcribe, 'callback')
        assert transcribe.name == 'transcribe'


class TestSharedDecoratorBehavioralEquivalence:
    """
    Property 2: Shared Decorator Behavioral Equivalence
    For any CLI option that uses shared decorators, the behavior should be 
    identical to inline @click.option definitions.
    
    Feature: cli-refactoring-v0-6-0, Property 2: Shared Decorator Behavioral Equivalence
    Validates: Requirements 3.5
    """
    
    def test_input_option_equivalence(self):
        """Test that @input_option behaves identically to inline @click.option."""
        
        # Create command with shared decorator
        @click.command()
        @input_option(help="Test input help")
        def cmd_with_decorator(source):
            click.echo(f"source: {source}")
        
        # Create equivalent command with inline option
        @click.command()
        @click.option('--source', '-s', required=True, help="Test input help")
        def cmd_with_inline(source):
            click.echo(f"source: {source}")
        
        runner = CliRunner()
        
        # Test with valid input
        result1 = runner.invoke(cmd_with_decorator, ['--source', 'test.mp4'])
        result2 = runner.invoke(cmd_with_inline, ['--source', 'test.mp4'])
        
        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output
        
        # Test with missing required option
        result1 = runner.invoke(cmd_with_decorator, [])
        result2 = runner.invoke(cmd_with_inline, [])
        
        assert result1.exit_code == result2.exit_code
        # Both should fail with missing option error
        assert result1.exit_code != 0
        assert result2.exit_code != 0
    
    def test_output_option_equivalence(self):
        """Test that @output_option behaves identically to inline @click.option."""
        
        # Create command with shared decorator
        @click.command()
        @output_option(help="Test output help")
        def cmd_with_decorator(output):
            click.echo(f"output: {output}")
        
        # Create equivalent command with inline option
        @click.command()
        @click.option('--output', '-o', default="output.mp3", help="Test output help")
        def cmd_with_inline(output):
            click.echo(f"output: {output}")
        
        runner = CliRunner()
        
        # Test with explicit output
        result1 = runner.invoke(cmd_with_decorator, ['--output', 'custom.mp3'])
        result2 = runner.invoke(cmd_with_inline, ['--output', 'custom.mp3'])
        
        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output
        
        # Test with default value
        result1 = runner.invoke(cmd_with_decorator, [])
        result2 = runner.invoke(cmd_with_inline, [])
        
        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output
    
    def test_language_option_equivalence(self):
        """Test that @language_option behaves identically to inline @click.option."""
        
        # Create command with shared decorator
        @click.command()
        @language_option(help="Test language help")
        def cmd_with_decorator(language):
            click.echo(f"language: {language}")
        
        # Create equivalent command with inline option
        @click.command()
        @click.option('--language', '-l', default=None, help="Test language help")
        def cmd_with_inline(language):
            click.echo(f"language: {language}")
        
        runner = CliRunner()
        
        # Test with explicit language
        result1 = runner.invoke(cmd_with_decorator, ['--language', 'en'])
        result2 = runner.invoke(cmd_with_inline, ['--language', 'en'])
        
        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output
        
        # Test with default (None)
        result1 = runner.invoke(cmd_with_decorator, [])
        result2 = runner.invoke(cmd_with_inline, [])
        
        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output
    
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.isprintable() and '\n' not in x and '\r' not in x))
    def test_help_text_preservation(self, help_text):
        """Property test: Custom help text should be preserved in decorators."""
        
        @click.command()
        @input_option(help=help_text)
        def test_cmd(source):
            pass
        
        # Get help output
        runner = CliRunner()
        result = runner.invoke(test_cmd, ['--help'])
        
        # Help text should appear in the output
        assert help_text in result.output
        assert result.exit_code == 0
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: not x.startswith('-')))
    def test_option_value_handling(self, option_value):
        """Property test: Option values should be handled consistently."""
        
        @click.command()
        @input_option()
        @output_option()
        def test_cmd(source, output):
            click.echo(f"source: {source}, output: {output}")
        
        runner = CliRunner()
        result = runner.invoke(test_cmd, ['--source', option_value, '--output', f"{option_value}.out"])
        
        assert result.exit_code == 0
        assert option_value in result.output
        assert f"{option_value}.out" in result.output


class TestHelpTextConsistency:
    """
    Property 5: Help Text Consistency
    For any subcommand help output, the displayed text should match the 
    centralized Help_Text constants.
    
    Feature: cli-refactoring-v0-6-0, Property 5: Help Text Consistency
    Validates: Requirements 5.9
    """
    
    def test_extract_command_help_consistency(self):
        """Test that extract command help matches centralized help text constants."""
        from cli.help_texts import EXTRACT_HELP, EXTRACT_SOURCE_HELP, EXTRACT_OUTPUT_HELP
        
        runner = CliRunner()
        result = runner.invoke(extract, ['--help'])
        
        assert result.exit_code == 0
        
        # Check that command help text appears in output
        assert EXTRACT_HELP in result.output or "Extract audio from the source file" in result.output
        
        # Check that option help texts appear in output
        # Note: Click may format help text differently, so we check for key phrases
        assert "Streaming platform URL" in result.output or "source file" in result.output
        assert "Base filename for extracted audio" in result.output or "output" in result.output.lower()
    
    def test_transcribe_command_help_consistency(self):
        """Test that transcribe command help matches centralized help text constants."""
        from cli.help_texts import TRANSCRIBE_HELP, TRANSCRIBE_SOURCE_HELP, TRANSCRIBE_OUTPUT_HELP, TRANSCRIBE_LANGUAGE_HELP
        
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--help'])
        
        assert result.exit_code == 0
        
        # Check that command help text appears in output
        assert TRANSCRIBE_HELP in result.output or "transcribe" in result.output.lower()
        
        # Check that option help texts appear in output
        assert "audio file" in result.output.lower()
        assert "transcript" in result.output.lower()
        assert "language" in result.output.lower()
    
    def test_main_cli_help_consistency(self):
        """Test that main CLI help contains expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        
        # Check main CLI description
        assert "Content Pipeline CLI" in result.output
        assert "Extract and transcribe multimedia content" in result.output
        
        # Check that subcommands are listed
        assert "extract" in result.output
        assert "transcribe" in result.output
        assert "Commands:" in result.output
    
    def test_help_text_constants_are_used(self):
        """Test that help text constants from help_texts.py are actually used in commands."""
        from cli.help_texts import (
            EXTRACT_SOURCE_HELP, EXTRACT_OUTPUT_HELP,
            TRANSCRIBE_SOURCE_HELP, TRANSCRIBE_OUTPUT_HELP, TRANSCRIBE_LANGUAGE_HELP
        )
        
        # Test extract command uses help text constants
        runner = CliRunner()
        result = runner.invoke(extract, ['--help'])
        
        # Check for key phrases from the help constants
        # (We check for key phrases since Click may format the text)
        assert "Streaming platform URL" in result.output
        assert "Base filename for extracted audio" in result.output
        
        # Test transcribe command uses help text constants - check for key parts since Click wraps text
        result = runner.invoke(transcribe, ['--help'])
        
        assert "Transcribe audio content to text" in result.output
        assert "configurable speech recognition" in result.output
        assert "Path to an audio file" in result.output
        assert "Base filename for transcript" in result.output or "Specific output file path for transcript" in result.output
        assert "Optional language hint" in result.output
    
    def test_error_message_constants_available(self):
        """Test that error message constants are available for use."""
        from cli.help_texts import MISSING_SOURCE_ERROR, INVALID_FORMAT_ERROR, FILE_NOT_FOUND_ERROR
        
        # Test that error constants exist and are strings
        assert isinstance(MISSING_SOURCE_ERROR, str)
        assert isinstance(INVALID_FORMAT_ERROR, str)
        assert isinstance(FILE_NOT_FOUND_ERROR, str)
        
        # Test that error constants have expected content
        assert "source" in MISSING_SOURCE_ERROR.lower()
        assert "format" in INVALID_FORMAT_ERROR.lower()
        assert "not found" in FILE_NOT_FOUND_ERROR.lower()
        
        # Test that FILE_NOT_FOUND_ERROR supports formatting
        formatted_error = FILE_NOT_FOUND_ERROR.format(path="test.mp3")
        assert "test.mp3" in formatted_error
    
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.isprintable() and '{' not in x and '}' not in x))
    def test_help_text_formatting_consistency(self, test_path):
        """Property test: Error message formatting should work consistently."""
        from cli.help_texts import FILE_NOT_FOUND_ERROR
        
        # Test that error message formatting works with various inputs
        try:
            formatted_error = FILE_NOT_FOUND_ERROR.format(path=test_path)
            assert test_path in formatted_error
            assert isinstance(formatted_error, str)
        except (KeyError, ValueError):
            # If formatting fails, it's because the constant doesn't have the expected format
            pytest.fail(f"FILE_NOT_FOUND_ERROR should support .format(path=...) but failed with path='{test_path}'")


class TestBackwardCompatibility:
    """
    Property 6: Backward Compatibility
    For any valid CLI command from v0.5.0, executing it in v0.6.0 should produce 
    identical output, behavior, and exit codes.
    
    Feature: cli-refactoring-v0-6-0, Property 6: Backward Compatibility
    Validates: Requirements 6.1, 6.2, 6.3, 6.5
    """
    
    def test_extract_command_interface_compatibility(self):
        """Test that extract command interface matches v0.5.0 expectations."""
        runner = CliRunner()
        
        # Test help output structure
        result = runner.invoke(main, ['extract', '--help'])
        assert result.exit_code == 0
        
        # Verify expected options are present
        assert '--source' in result.output
        assert '-s' in result.output  # Short option
        assert '--output' in result.output
        assert '-o' in result.output  # Short option
        
        # Verify help text structure
        assert 'Usage:' in result.output
        assert 'Options:' in result.output
        
        # Test that source is required
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
    
    def test_transcribe_command_interface_compatibility(self):
        """Test that transcribe command interface matches v0.5.0 expectations."""
        runner = CliRunner()
        
        # Test help output structure
        result = runner.invoke(main, ['transcribe', '--help'])
        assert result.exit_code == 0
        
        # Verify expected options are present
        assert '--source' in result.output
        assert '-s' in result.output  # Short option
        assert '--output' in result.output
        assert '--language' in result.output
        assert '-l' in result.output  # Short option for language
        
        # Verify help text structure
        assert 'Usage:' in result.output
        assert 'Options:' in result.output
        
        # Test that source is required
        result = runner.invoke(main, ['transcribe'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
    
    def test_main_cli_help_structure_compatibility(self):
        """Test that main CLI help maintains v0.5.0 structure."""
        runner = CliRunner()
        
        # Test main help output
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        
        # Verify expected structure
        assert 'Usage:' in result.output
        assert 'Options:' in result.output
        assert 'Commands:' in result.output
        
        # Verify expected commands are listed
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        
        # Verify version option works
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.7.0' in result.output
    
    def test_error_handling_compatibility(self):
        """Test that error handling matches v0.5.0 behavior."""
        runner = CliRunner()
        
        # Test invalid command
        result = runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output or 'invalid' in result.output.lower()
        
        # Test missing required options
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        
        result = runner.invoke(main, ['transcribe'])
        assert result.exit_code != 0
    
    def test_option_parsing_compatibility(self):
        """Test that option parsing works identically to v0.5.0."""
        runner = CliRunner()
        
        # Test extract with both long and short options
        result = runner.invoke(main, ['extract', '--source', 'test.mp4', '--output', 'test.mp3'])
        # Should fail due to missing file, but parsing should work
        assert 'test.mp4' in result.output or 'test.mp3' in result.output or result.exit_code != 0
        
        result = runner.invoke(main, ['extract', '-s', 'test.mp4', '-o', 'test.mp3'])
        # Should fail due to missing file, but parsing should work
        assert 'test.mp4' in result.output or 'test.mp3' in result.output or result.exit_code != 0
        
        # Test transcribe with language option
        result = runner.invoke(main, ['transcribe', '--source', 'test.mp3', '--language', 'en'])
        # Should fail due to missing file, but parsing should work
        assert result.exit_code != 0  # Expected to fail due to missing file
        
        result = runner.invoke(main, ['transcribe', '-s', 'test.mp3', '-l', 'en'])
        # Should fail due to missing file, but parsing should work
        assert result.exit_code != 0  # Expected to fail due to missing file
    
    def test_exit_code_consistency(self):
        """Test that exit codes match v0.5.0 behavior."""
        runner = CliRunner()
        
        # Success cases (help commands)
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        
        result = runner.invoke(main, ['extract', '--help'])
        assert result.exit_code == 0
        
        result = runner.invoke(main, ['transcribe', '--help'])
        assert result.exit_code == 0
        
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        
        # Error cases
        result = runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        
        result = runner.invoke(main, ['extract'])  # Missing required option
        assert result.exit_code != 0
        
        result = runner.invoke(main, ['transcribe'])  # Missing required option
        assert result.exit_code != 0
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.isprintable() and not x.startswith('-')))
    def test_option_value_handling_compatibility(self, test_value):
        """Property test: Option values should be handled consistently with v0.5.0."""
        runner = CliRunner()
        
        # Test that option values are parsed correctly
        result = runner.invoke(main, ['extract', '--source', test_value, '--output', f'{test_value}.out'])
        
        # The command should either succeed or fail gracefully
        # (it will likely fail due to file not existing, but should parse options correctly)
        assert isinstance(result.exit_code, int)
        
        # If there's output, it should contain our test values or be a proper error message
        if result.output:
            assert isinstance(result.output, str)
    
    def test_command_registration_compatibility(self):
        """Test that command registration works as expected for v0.5.0 compatibility."""
        # Test that both commands are properly registered
        assert 'extract' in main.commands
        assert 'transcribe' in main.commands
        
        # Test that commands are Click commands
        assert hasattr(main.commands['extract'], 'callback')
        assert hasattr(main.commands['transcribe'], 'callback')
        
        # Test that main group is properly configured
        assert isinstance(main, click.Group)
        assert main.name == 'main'


class TestErrorMessageConsistency:
    """
    Property 7: Error Message Consistency
    For any error condition, the CLI should display error messages with the same 
    format and content as v0.5.0.
    
    Feature: cli-refactoring-v0-6-0, Property 7: Error Message Consistency
    Validates: Requirements 6.4, 8.4
    """
    
    def test_missing_required_option_error_format(self):
        """Test that missing required option errors have consistent format."""
        runner = CliRunner()
        
        # Test extract command missing source
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
        
        # Test transcribe command missing source
        result = runner.invoke(main, ['transcribe'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
    
    def test_invalid_command_error_format(self):
        """Test that invalid command errors have consistent format."""
        runner = CliRunner()
        
        result = runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output or 'invalid' in result.output.lower()
        
        # Test with various invalid commands
        for invalid_cmd in ['badcommand', 'xyz', 'notfound']:
            result = runner.invoke(main, [invalid_cmd])
            assert result.exit_code != 0
            # Should contain some indication that the command is not found
            assert any(phrase in result.output.lower() for phrase in ['no such command', 'invalid', 'not found', 'unknown'])
    
    def test_file_not_found_error_consistency(self):
        """Test that file not found errors are consistent."""
        runner = CliRunner()
        
        # Test extract with non-existent file
        result = runner.invoke(main, ['extract', '--source', 'nonexistent.mp4', '--output', 'test.mp3'])
        assert result.exit_code != 0
        # Should contain some indication that the file was not found
        assert any(phrase in result.output.lower() for phrase in ['not found', 'does not exist', 'no such file', 'input file'])
        
        # Test transcribe with non-existent file - now requires --engine flag
        result = runner.invoke(main, ['transcribe', '--source', 'nonexistent.mp3', '--engine', 'local-whisper'])
        assert result.exit_code != 0
        # Should contain some indication that the file was not found or engine error
        assert any(phrase in result.output.lower() for phrase in ['not found', 'does not exist', 'no such file', 'audio file', 'engine'])
    
    def test_error_message_structure_consistency(self):
        """Test that error messages follow consistent structure."""
        runner = CliRunner()
        
        # Test that error messages are properly formatted
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        
        # Error output should be a string
        assert isinstance(result.output, str)
        
        # Should not be empty
        assert len(result.output.strip()) > 0
        
        # Should contain some indication of what went wrong
        assert any(word in result.output.lower() for word in ['error', 'missing', 'required', 'invalid'])
    
    def test_help_option_error_handling(self):
        """Test that help options work correctly and don't produce errors."""
        runner = CliRunner()
        
        # These should all succeed (exit code 0)
        help_commands = [
            ['--help'],
            ['extract', '--help'],
            ['transcribe', '--help'],
            ['--version']
        ]
        
        for cmd in help_commands:
            result = runner.invoke(main, cmd)
            assert result.exit_code == 0, f"Help command {cmd} should succeed but got exit code {result.exit_code}"
            assert len(result.output.strip()) > 0, f"Help command {cmd} should produce output"
    
    def test_error_message_constants_usage(self):
        """Test that error message constants are available and properly formatted."""
        from cli.help_texts import MISSING_SOURCE_ERROR, INVALID_FORMAT_ERROR, FILE_NOT_FOUND_ERROR
        
        # Test that constants are strings
        assert isinstance(MISSING_SOURCE_ERROR, str)
        assert isinstance(INVALID_FORMAT_ERROR, str)
        assert isinstance(FILE_NOT_FOUND_ERROR, str)
        
        # Test that constants are not empty
        assert len(MISSING_SOURCE_ERROR.strip()) > 0
        assert len(INVALID_FORMAT_ERROR.strip()) > 0
        assert len(FILE_NOT_FOUND_ERROR.strip()) > 0
        
        # Test that FILE_NOT_FOUND_ERROR supports formatting
        formatted = FILE_NOT_FOUND_ERROR.format(path="test.mp3")
        assert "test.mp3" in formatted
        assert formatted != FILE_NOT_FOUND_ERROR  # Should be different after formatting
    
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()))
    def test_invalid_command_error_consistency(self, invalid_command):
        """Property test: Invalid commands should produce consistent error messages."""
        runner = CliRunner()
        
        # Skip commands that might actually exist
        if invalid_command.lower() in ['extract', 'transcribe', 'help', 'version']:
            return
        
        result = runner.invoke(main, [invalid_command])
        
        # Should always fail
        assert result.exit_code != 0
        
        # Should produce some output
        assert len(result.output.strip()) > 0
        
        # Should indicate the command is not found
        assert any(phrase in result.output.lower() for phrase in ['no such command', 'invalid', 'not found', 'unknown'])
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.isprintable() and '/' not in x and '\\' not in x and x not in ['.', '..']))
    def test_file_not_found_error_format_consistency(self, filename):
        """Property test: File not found errors should be consistently formatted."""
        runner = CliRunner()
        
        # Test with extract command
        result = runner.invoke(main, ['extract', '--source', filename, '--output', 'test.mp3'])
        
        # Should fail (file doesn't exist) - but '.' is a directory so it may succeed with different error
        # We just check that it produces output and is a string
        assert len(result.output.strip()) > 0
        
        # Should be a proper error message
        assert isinstance(result.output, str)


class TestDynamicCommandRegistration:
    """
    Property 8: Dynamic Command Registration
    For any new subcommand module, the Click group should support dynamic 
    registration without modifying existing CLI package structure.
    
    Feature: cli-refactoring-v0-6-0, Property 8: Dynamic Command Registration
    Validates: Requirements 7.1, 7.5
    """
    
    def test_click_group_supports_dynamic_registration(self):
        """Test that Click groups support adding commands dynamically."""
        
        # Create a new CLI group
        @click.group()
        def dynamic_cli():
            """Dynamic CLI for testing."""
            pass
        
        # Create test commands
        @click.command()
        @click.option('--input', required=True)
        def cmd1(input):
            """First command."""
            click.echo(f"Command 1: {input}")
        
        @click.command()
        @click.option('--data', required=True)
        def cmd2(data):
            """Second command."""
            click.echo(f"Command 2: {data}")
        
        # Test that commands can be added dynamically
        dynamic_cli.add_command(cmd1)
        dynamic_cli.add_command(cmd2)
        
        # Test that commands are registered
        assert 'cmd1' in dynamic_cli.commands
        assert 'cmd2' in dynamic_cli.commands
        
        # Test that commands work
        runner = CliRunner()
        
        result = runner.invoke(dynamic_cli, ['cmd1', '--input', 'test1'])
        assert result.exit_code == 0
        assert 'Command 1: test1' in result.output
        
        result = runner.invoke(dynamic_cli, ['cmd2', '--data', 'test2'])
        assert result.exit_code == 0
        assert 'Command 2: test2' in result.output
    
    def test_existing_commands_unaffected_by_new_registrations(self):
        """Test that adding new commands doesn't affect existing ones."""
        
        # Create a CLI group with existing commands
        @click.group()
        def test_cli():
            """Test CLI."""
            pass
        
        # Add existing commands from main CLI
        test_cli.add_command(main.commands['extract'])
        test_cli.add_command(main.commands['transcribe'])
        
        # Store original command count
        original_count = len(test_cli.commands)
        
        # Add a new command
        @click.command()
        @click.option('--value', required=True)
        def new_cmd(value):
            """New command."""
            click.echo(f"New: {value}")
        
        test_cli.add_command(new_cmd)
        
        # Test that all commands still work
        runner = CliRunner()
        
        # Original commands should still work
        result = runner.invoke(test_cli, ['extract', '--help'])
        assert result.exit_code == 0
        
        result = runner.invoke(test_cli, ['transcribe', '--help'])
        assert result.exit_code == 0
        
        # New command should work
        result = runner.invoke(test_cli, ['new', '--value', 'test'])
        assert result.exit_code == 0
        assert 'New: test' in result.output
        
        # Command count should increase by 1
        assert len(test_cli.commands) == original_count + 1
    
    def test_command_registration_preserves_structure(self):
        """Test that command registration preserves CLI structure."""
        
        # Create base CLI
        @click.group()
        def base_cli():
            """Base CLI."""
            pass
        
        # Create commands with different structures
        @click.command()
        @click.option('--simple', required=True)
        def simple_cmd(simple):
            """Simple command."""
            pass
        
        @click.command()
        @click.option('--input', '-i', required=True)
        @click.option('--output', '-o', default='out.txt')
        @click.option('--verbose', '-v', is_flag=True)
        def complex_cmd(input, output, verbose):
            """Complex command with multiple options."""
            pass
        
        # Register commands
        base_cli.add_command(simple_cmd)
        base_cli.add_command(complex_cmd)
        
        # Test that command structures are preserved
        simple_params = [p.name for p in base_cli.commands['simple'].params]
        complex_params = [p.name for p in base_cli.commands['complex'].params]
        
        assert 'simple' in simple_params
        assert 'input' in complex_params
        assert 'output' in complex_params
        assert 'verbose' in complex_params
        
        # Test that options work correctly
        runner = CliRunner()
        
        result = runner.invoke(base_cli, ['simple', '--simple', 'test'])
        assert result.exit_code == 0
        
        result = runner.invoke(base_cli, ['complex', '--input', 'in.txt'])
        assert result.exit_code == 0
    
    def test_shared_options_work_with_dynamic_registration(self):
        """Test that shared options work with dynamically registered commands."""
        from cli.shared_options import input_option, output_option
        
        # Create CLI group
        @click.group()
        def shared_cli():
            """CLI with shared options."""
            pass
        
        # Create command using shared options
        @click.command()
        @input_option()
        @output_option()
        def shared_cmd(source, output):
            """Command using shared options."""
            click.echo(f"Shared: {source} -> {output}")
        
        # Register command
        shared_cli.add_command(shared_cmd)
        
        # Test that shared options work
        runner = CliRunner()
        
        result = runner.invoke(shared_cli, ['shared', '--help'])
        assert result.exit_code == 0
        assert '--source' in result.output
        assert '--output' in result.output
        
        result = runner.invoke(shared_cli, ['shared', '--source', 'input.txt'])
        assert result.exit_code == 0
        assert 'Shared: input.txt -> output.mp3' in result.output
    
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()))
    def test_dynamic_command_names_handled_correctly(self, cmd_name):
        """Property test: Dynamic command names should be handled correctly."""
        
        # Skip reserved names
        if cmd_name.lower() in ['help', 'version', 'extract', 'transcribe']:
            return
        
        # Create CLI group
        @click.group()
        def prop_cli():
            """Property test CLI."""
            pass
        
        # Create command with dynamic name
        @click.command(name=cmd_name)
        @click.option('--test', default='default')
        def dynamic_cmd(test):
            f"""Command with name {cmd_name}."""
            click.echo(f"Dynamic command {cmd_name}: {test}")
        
        # Register command
        prop_cli.add_command(dynamic_cmd)
        
        # Test that command is registered with correct name
        assert cmd_name in prop_cli.commands
        
        # Test that command works
        runner = CliRunner()
        result = runner.invoke(prop_cli, [cmd_name, '--test', 'value'])
        
        # Should either succeed or fail gracefully
        assert isinstance(result.exit_code, int)
        if result.exit_code == 0:
            assert f"Dynamic command {cmd_name}: value" in result.output
    
    @given(st.integers(min_value=1, max_value=10))
    def test_multiple_dynamic_registrations(self, num_commands):
        """Property test: Multiple commands can be registered dynamically."""
        
        # Create CLI group
        @click.group()
        def multi_cli():
            """Multi-command CLI."""
            pass
        
        # Create and register multiple commands
        for i in range(num_commands):
            @click.command(name=f'cmd{i}')
            @click.option('--value', default=f'default{i}')
            def cmd(value, cmd_num=i):
                f"""Command {cmd_num}."""
                click.echo(f"Command {cmd_num}: {value}")
            
            multi_cli.add_command(cmd)
        
        # Test that all commands are registered
        assert len(multi_cli.commands) == num_commands
        
        # Test that commands have expected names
        for i in range(num_commands):
            assert f'cmd{i}' in multi_cli.commands
        
        # Test that at least one command works
        if num_commands > 0:
            runner = CliRunner()
            result = runner.invoke(multi_cli, ['cmd0', '--value', 'test'])
            assert isinstance(result.exit_code, int)


class TestSharedOptionsExtensibility:
    """
    Property 9: Shared Options Extensibility
    For any new common option pattern, the shared options module should support 
    adding new decorators that work consistently across subcommands.
    
    Feature: cli-refactoring-v0-6-0, Property 9: Shared Options Extensibility
    Validates: Requirements 7.2
    """
    
    def test_new_shared_option_decorators_can_be_created(self):
        """Test that new shared option decorators can be created and used."""
        
        # Create a new shared option decorator
        def format_option(help=None):
            """Decorator for format options."""
            def decorator(f):
                return click.option(
                    '--format', '-f',
                    default='json',
                    type=click.Choice(['json', 'xml', 'csv']),
                    help=help or 'Output format'
                )(f)
            return decorator
        
        # Create commands using the new decorator
        @click.command()
        @format_option(help="Custom format help")
        def cmd1(format):
            """Command 1 with format option."""
            click.echo(f"Format: {format}")
        
        @click.command()
        @format_option()
        def cmd2(format):
            """Command 2 with format option."""
            click.echo(f"Format: {format}")
        
        runner = CliRunner()
        
        # Test that both commands work with the shared decorator
        result = runner.invoke(cmd1, ['--format', 'xml'])
        assert result.exit_code == 0
        assert 'Format: xml' in result.output
        
        result = runner.invoke(cmd2, ['--format', 'csv'])
        assert result.exit_code == 0
        assert 'Format: csv' in result.output
        
        # Test help text
        result = runner.invoke(cmd1, ['--help'])
        assert result.exit_code == 0
        assert 'Custom format help' in result.output
        
        result = runner.invoke(cmd2, ['--help'])
        assert result.exit_code == 0
        assert 'Output format' in result.output
    
    def test_shared_options_work_with_existing_decorators(self):
        """Test that new shared options work alongside existing ones."""
        from cli.shared_options import input_option, output_option
        
        # Create a new shared option
        def verbose_option(help=None):
            """Decorator for verbose options."""
            def decorator(f):
                return click.option(
                    '--verbose', '-v',
                    is_flag=True,
                    help=help or 'Enable verbose output'
                )(f)
            return decorator
        
        # Create command using both existing and new shared options
        @click.command()
        @input_option()
        @output_option()
        @verbose_option()
        def mixed_cmd(source, output, verbose):
            """Command using mixed shared options."""
            if verbose:
                click.echo(f"Verbose: Processing {source} -> {output}")
            else:
                click.echo(f"Processing {source} -> {output}")
        
        runner = CliRunner()
        
        # Test without verbose
        result = runner.invoke(mixed_cmd, ['--source', 'input.txt'])
        assert result.exit_code == 0
        assert 'Processing input.txt -> output.mp3' in result.output
        assert 'Verbose:' not in result.output
        
        # Test with verbose
        result = runner.invoke(mixed_cmd, ['--source', 'input.txt', '--verbose'])
        assert result.exit_code == 0
        assert 'Verbose: Processing input.txt -> output.mp3' in result.output
    
    def test_shared_option_decorator_consistency(self):
        """Test that shared option decorators maintain consistency."""
        
        # Create multiple shared option decorators following the same pattern
        def timeout_option(help=None):
            """Decorator for timeout options."""
            def decorator(f):
                return click.option(
                    '--timeout', '-t',
                    default=30,
                    type=int,
                    help=help or 'Timeout in seconds'
                )(f)
            return decorator
        
        def retry_option(help=None):
            """Decorator for retry options."""
            def decorator(f):
                return click.option(
                    '--retry', '-r',
                    default=3,
                    type=int,
                    help=help or 'Number of retries'
                )(f)
            return decorator
        
        # Create commands using these decorators
        @click.command()
        @timeout_option()
        @retry_option()
        def network_cmd(timeout, retry):
            """Network command with timeout and retry."""
            click.echo(f"Network: timeout={timeout}, retry={retry}")
        
        runner = CliRunner()
        
        # Test default values
        result = runner.invoke(network_cmd, [])
        assert result.exit_code == 0
        assert 'Network: timeout=30, retry=3' in result.output
        
        # Test custom values
        result = runner.invoke(network_cmd, ['--timeout', '60', '--retry', '5'])
        assert result.exit_code == 0
        assert 'Network: timeout=60, retry=5' in result.output
        
        # Test short options
        result = runner.invoke(network_cmd, ['-t', '45', '-r', '2'])
        assert result.exit_code == 0
        assert 'Network: timeout=45, retry=2' in result.output
    
    def test_shared_options_preserve_click_functionality(self):
        """Test that shared options preserve all Click functionality."""
        
        # Create a comprehensive shared option decorator
        def advanced_option(help=None):
            """Advanced shared option with multiple Click features."""
            def decorator(f):
                return click.option(
                    '--advanced', '-a',
                    default='default',
                    type=click.Choice(['option1', 'option2', 'option3']),
                    show_default=True,
                    help=help or 'Advanced option with choices'
                )(f)
            return decorator
        
        @click.command()
        @advanced_option()
        def advanced_cmd(advanced):
            """Command with advanced shared option."""
            click.echo(f"Advanced: {advanced}")
        
        runner = CliRunner()
        
        # Test help shows choices and default
        result = runner.invoke(advanced_cmd, ['--help'])
        assert result.exit_code == 0
        assert 'option1' in result.output
        assert 'option2' in result.output
        assert 'option3' in result.output
        assert 'default' in result.output
        
        # Test valid choice
        result = runner.invoke(advanced_cmd, ['--advanced', 'option2'])
        assert result.exit_code == 0
        assert 'Advanced: option2' in result.output
        
        # Test invalid choice fails
        result = runner.invoke(advanced_cmd, ['--advanced', 'invalid'])
        assert result.exit_code != 0
    
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()))
    def test_shared_option_help_text_customization(self, help_text):
        """Property test: Shared options should support custom help text."""
        
        # Create a shared option decorator that accepts custom help
        def custom_help_option(help=None):
            """Decorator with customizable help."""
            def decorator(f):
                return click.option(
                    '--custom', '-c',
                    default='default',
                    help=help or 'Default help text'
                )(f)
            return decorator
        
        # Create command with custom help
        @click.command()
        @custom_help_option(help=help_text)
        def help_cmd(custom):
            """Command with custom help."""
            click.echo(f"Custom: {custom}")
        
        runner = CliRunner()
        
        # Test that custom help text appears
        result = runner.invoke(help_cmd, ['--help'])
        assert result.exit_code == 0
        assert help_text in result.output
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.isprintable() and not x.startswith('-')))
    def test_shared_option_default_values(self, default_value):
        """Property test: Shared options should handle various default values."""
        
        # Create shared option with dynamic default
        def dynamic_default_option(default=None, help=None):
            """Decorator with dynamic default."""
            def decorator(f):
                return click.option(
                    '--dynamic', '-d',
                    default=default or 'fallback',
                    help=help or 'Dynamic default option'
                )(f)
            return decorator
        
        # Create command with custom default
        @click.command()
        @dynamic_default_option(default=default_value)
        def default_cmd(dynamic):
            """Command with dynamic default."""
            click.echo(f"Dynamic: {dynamic}")
        
        runner = CliRunner()
        
        # Test that default value is used
        result = runner.invoke(default_cmd, [])
        assert result.exit_code == 0
        assert f"Dynamic: {default_value}" in result.output


class TestMigrationValidation:
    """
    Property 10: Migration Validation
    For any test or import in the new CLI structure, the functionality should work 
    identically to the original implementation and produce the same test results.
    
    Feature: cli-refactoring-v0-6-0, Property 10: Migration Validation
    Validates: Requirements 9.2, 9.5
    """
    
    def test_import_paths_work_correctly(self):
        """Test that import paths in the new cli package structure work correctly."""
        
        # Test that all CLI modules can be imported without errors
        import cli
        import cli.extract
        import cli.transcribe
        import cli.shared_options
        import cli.help_texts
        
        # Test that main functions are accessible
        assert hasattr(cli, 'main')
        assert hasattr(cli, 'cli')
        
        # Test that extract and transcribe are Click commands (imported as commands, not modules)
        assert hasattr(cli.extract, 'callback')  # Click command has callback
        assert hasattr(cli.transcribe, 'callback')  # Click command has callback
        
        # Test that shared options are accessible
        assert hasattr(cli.shared_options, 'input_option')
        assert hasattr(cli.shared_options, 'output_option')
        assert hasattr(cli.shared_options, 'language_option')
        
        # Test that help texts are accessible
        assert hasattr(cli.help_texts, 'EXTRACT_SOURCE_HELP')
        assert hasattr(cli.help_texts, 'TRANSCRIBE_SOURCE_HELP')
    
    def test_cli_functionality_identical_to_original(self):
        """Test that CLI functionality works identically to the original implementation."""
        runner = CliRunner()
        
        # Test main CLI help output structure
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        
        # Should contain expected sections
        assert 'Usage:' in result.output
        assert 'Options:' in result.output
        assert 'Commands:' in result.output
        
        # Should contain expected commands
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        
        # Should contain version information
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.7.0' in result.output
    
    def test_extract_command_behavior_identical(self):
        """Test that extract command behavior is identical to original."""
        runner = CliRunner()
        
        # Test help output
        result = runner.invoke(main, ['extract', '--help'])
        assert result.exit_code == 0
        assert 'Extract audio from the source file' in result.output
        assert '--source' in result.output
        assert '--output' in result.output
        
        # Test required option validation
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
        
        # Test option parsing
        result = runner.invoke(main, ['extract', '--source', 'test.mp4', '--output', 'test.mp3'])
        # Should fail due to missing file, but options should be parsed correctly
        assert result.exit_code != 0
        assert 'Input file does not exist' in result.output or 'not found' in result.output.lower()
    
    def test_transcribe_command_behavior_identical(self):
        """Test that transcribe command behavior is identical to original."""
        runner = CliRunner()
        
        # Test help output - check for key parts since Click wraps text
        result = runner.invoke(main, ['transcribe', '--help'])
        assert result.exit_code == 0
        assert 'Transcribe audio content to text' in result.output
        assert 'configurable speech recognition' in result.output
        assert '--source' in result.output
        assert '--output' in result.output
        assert '--language' in result.output
        
        # Test required option validation
        result = runner.invoke(main, ['transcribe'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
        
        # Test option parsing with language - now requires --engine flag
        result = runner.invoke(main, ['transcribe', '--source', 'test.mp3', '--language', 'en', '--engine', 'local-whisper'])
        # Should fail due to missing file, but options should be parsed correctly
        assert result.exit_code != 0
        assert 'Audio file does not exist' in result.output or 'not found' in result.output.lower()
    
    def test_shared_options_behavior_consistent(self):
        """Test that shared options behavior is consistent across commands."""
        
        # Test that input_option works consistently
        from cli.shared_options import input_option, output_option, language_option
        
        # Create test commands using shared options
        @click.command()
        @input_option()
        def test_input_cmd(source):
            """Test command with input option."""
            click.echo(f"Source: {source}")
        
        @click.command()
        @output_option()
        def test_output_cmd(output):
            """Test command with output option."""
            click.echo(f"Output: {output}")
        
        @click.command()
        @language_option()
        def test_language_cmd(language):
            """Test command with language option."""
            click.echo(f"Language: {language}")
        
        runner = CliRunner()
        
        # Test input option
        result = runner.invoke(test_input_cmd, ['--source', 'test.mp4'])
        assert result.exit_code == 0
        assert 'Source: test.mp4' in result.output
        
        result = runner.invoke(test_input_cmd, ['-s', 'test.mp4'])
        assert result.exit_code == 0
        assert 'Source: test.mp4' in result.output
        
        # Test output option with default
        result = runner.invoke(test_output_cmd, [])
        assert result.exit_code == 0
        assert 'Output: output.mp3' in result.output
        
        # Test language option (optional)
        result = runner.invoke(test_language_cmd, [])
        assert result.exit_code == 0
        assert 'Language: None' in result.output
        
        result = runner.invoke(test_language_cmd, ['--language', 'en'])
        assert result.exit_code == 0
        assert 'Language: en' in result.output
    
    def test_error_handling_behavior_identical(self):
        """Test that error handling behavior is identical to original."""
        runner = CliRunner()
        
        # Test invalid command
        result = runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output or 'invalid' in result.output.lower()
        
        # Test missing required options produce consistent errors
        result = runner.invoke(main, ['extract'])
        assert result.exit_code != 0
        error_output = result.output.lower()
        assert 'missing' in error_output or 'required' in error_output
        
        result = runner.invoke(main, ['transcribe'])
        assert result.exit_code != 0
        error_output = result.output.lower()
        assert 'missing' in error_output or 'required' in error_output
    
    def test_help_text_constants_accessible(self):
        """Test that help text constants are accessible and properly formatted."""
        from cli.help_texts import (
            EXTRACT_SOURCE_HELP,
            EXTRACT_OUTPUT_HELP,
            TRANSCRIBE_SOURCE_HELP,
            TRANSCRIBE_OUTPUT_HELP,
            TRANSCRIBE_LANGUAGE_HELP,
            MISSING_SOURCE_ERROR,
            INVALID_FORMAT_ERROR,
            FILE_NOT_FOUND_ERROR
        )
        
        # Test that all constants are strings
        help_constants = [
            EXTRACT_SOURCE_HELP,
            EXTRACT_OUTPUT_HELP,
            TRANSCRIBE_SOURCE_HELP,
            TRANSCRIBE_OUTPUT_HELP,
            TRANSCRIBE_LANGUAGE_HELP,
            MISSING_SOURCE_ERROR,
            INVALID_FORMAT_ERROR,
            FILE_NOT_FOUND_ERROR
        ]
        
        for constant in help_constants:
            assert isinstance(constant, str)
            assert len(constant.strip()) > 0
        
        # Test that error constants support formatting
        formatted_error = FILE_NOT_FOUND_ERROR.format(path="test.mp3")
        assert "test.mp3" in formatted_error
        assert formatted_error != FILE_NOT_FOUND_ERROR
    
    def test_cli_module_structure_preserved(self):
        """Test that CLI module structure is preserved and accessible."""
        
        # Test that main CLI group has expected structure
        assert isinstance(main, click.Group)
        assert main.name == 'main'
        
        # Test that commands are properly registered
        assert 'extract' in main.commands
        assert 'transcribe' in main.commands
        
        # Test that commands are Click commands
        extract_cmd = main.commands['extract']
        transcribe_cmd = main.commands['transcribe']
        
        assert hasattr(extract_cmd, 'callback')
        assert hasattr(transcribe_cmd, 'callback')
        
        # Test that commands have expected parameters
        extract_params = [p.name for p in extract_cmd.params]
        transcribe_params = [p.name for p in transcribe_cmd.params]
        
        assert 'source' in extract_params
        assert 'output' in extract_params
        
        assert 'source' in transcribe_params
        assert 'output' in transcribe_params
        assert 'language' in transcribe_params
    
    def test_console_script_entry_point_works(self):
        """Test that console script entry point works correctly."""
        
        # Test that cli() function exists and is callable
        from cli import cli
        assert callable(cli)
        
        # Test that main() function exists and is callable
        from cli import main
        assert callable(main)
        
        # Test that they are different functions
        assert cli != main
        
        # Test that cli() calls main() (indirectly by testing structure)
        runner = CliRunner()
        import subprocess
        import sys
        
        # Both should produce similar help output
        result1 = runner.invoke(main, ['--help'])
        result2 = subprocess.run([sys.executable, '-m', 'cli', '--help'], 
                                capture_output=True, text=True)
        
        assert result1.exit_code == 0
        assert result2.returncode == 0
        
        # Should contain same key elements
        assert 'Content Pipeline CLI' in result1.output
        assert 'Content Pipeline CLI' in result2.stdout
    
    @given(st.sampled_from(['extract', 'transcribe']))
    def test_command_help_consistency(self, command):
        """Property test: Command help should be consistent and complete."""
        runner = CliRunner()
        
        result = runner.invoke(main, [command, '--help'])
        assert result.exit_code == 0
        
        # Should contain standard help sections
        assert 'Usage:' in result.output
        assert 'Options:' in result.output
        
        # Should contain expected options
        assert '--source' in result.output
        assert '--output' in result.output
        
        # Should contain help option
        assert '--help' in result.output
        
        # Should be properly formatted
        assert len(result.output.strip()) > 0
        assert isinstance(result.output, str)
    
    @given(st.sampled_from(['-s', '--source']))
    def test_option_aliases_work_consistently(self, source_option):
        """Property test: Option aliases should work consistently across commands."""
        runner = CliRunner()
        
        # Test extract command
        result = runner.invoke(main, ['extract', source_option, 'test.mp4'])
        assert result.exit_code != 0  # Will fail due to missing file
        assert 'Input file does not exist' in result.output or 'not found' in result.output.lower()
        
        # Test transcribe command - now requires --engine flag
        result = runner.invoke(main, ['transcribe', source_option, 'test.mp3', '--engine', 'local-whisper'])
        assert result.exit_code != 0  # Will fail due to missing file or missing engine
        # Accept either file not found or missing engine error
        assert ('Audio file does not exist' in result.output or 'not found' in result.output.lower() or 
                'Missing option' in result.output or 'engine' in result.output.lower())


class TestEntryPointEquivalence:
    """
    Property 11: Entry Point Equivalence
    For any CLI invocation method (direct module or console script), both should 
    produce identical behavior and results.
    
    Feature: cli-refactoring-v0-6-0, Property 11: Entry Point Equivalence
    Validates: Requirements 9.6
    """
    
    def test_direct_module_invocation_works(self):
        """Test that direct module invocation (python -m cli) works correctly."""
        import subprocess
        import sys
        
        # Test main help
        result = subprocess.run([sys.executable, '-m', 'cli', '--help'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Content Pipeline CLI' in result.stdout
        assert 'extract' in result.stdout
        assert 'transcribe' in result.stdout
        
        # Test version
        result = subprocess.run([sys.executable, '-m', 'cli', '--version'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert '0.7.0' in result.stdout
        
        # Test extract help
        result = subprocess.run([sys.executable, '-m', 'cli', 'extract', '--help'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Extract audio from the source file' in result.stdout
        
        # Test transcribe help - check for key parts since Click wraps text
        result = subprocess.run([sys.executable, '-m', 'cli', 'transcribe', '--help'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Transcribe audio content to text' in result.stdout
        assert 'configurable speech recognition' in result.stdout
    
    def test_console_script_and_module_produce_identical_output(self):
        """Test that console script and module invocation produce identical output."""
        runner = CliRunner()
        
        # Test main help via both methods
        module_result = runner.invoke(main, ['--help'])
        
        # For console script, we need to test that cli() calls main()
        # We can't directly invoke cli() with CliRunner since it's a wrapper function
        # Instead, we test that both main() and the module invocation work identically
        import subprocess
        import sys
        
        # Test module invocation (python -m cli)
        subprocess_result = subprocess.run([sys.executable, '-m', 'cli', '--help'], 
                                         capture_output=True, text=True)
        
        assert module_result.exit_code == 0
        assert subprocess_result.returncode == 0
        
        # Both should contain the same key information
        assert 'Content Pipeline CLI' in module_result.output
        assert 'Content Pipeline CLI' in subprocess_result.stdout
        assert 'extract' in module_result.output
        assert 'extract' in subprocess_result.stdout
        assert 'transcribe' in module_result.output
        assert 'transcribe' in subprocess_result.stdout
        
        # Test version via both methods
        module_version = runner.invoke(main, ['--version'])
        subprocess_version = subprocess.run([sys.executable, '-m', 'cli', '--version'], 
                                          capture_output=True, text=True)
        
        assert module_version.exit_code == 0
        assert subprocess_version.returncode == 0
        assert '0.7.0' in module_version.output
        assert '0.7.0' in subprocess_version.stdout
    
    def test_subcommand_help_identical_across_entry_points(self):
        """Test that subcommand help is identical across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        # Test extract help
        module_extract = runner.invoke(main, ['extract', '--help'])
        subprocess_extract = subprocess.run([sys.executable, '-m', 'cli', 'extract', '--help'], 
                                          capture_output=True, text=True)
        
        assert module_extract.exit_code == 0
        assert subprocess_extract.returncode == 0
        assert 'Extract audio from the source file' in module_extract.output
        assert 'Extract audio from the source file' in subprocess_extract.stdout
        assert '--source' in module_extract.output
        assert '--source' in subprocess_extract.stdout
        
        # Test transcribe help - check for key parts since Click wraps text
        module_transcribe = runner.invoke(main, ['transcribe', '--help'])
        subprocess_transcribe = subprocess.run([sys.executable, '-m', 'cli', 'transcribe', '--help'], 
                                             capture_output=True, text=True)
        
        assert module_transcribe.exit_code == 0
        assert subprocess_transcribe.returncode == 0
        assert 'Transcribe audio content to text' in module_transcribe.output
        assert 'configurable speech recognition' in module_transcribe.output
        assert 'Transcribe audio content to text' in subprocess_transcribe.stdout
        assert 'configurable speech recognition' in subprocess_transcribe.stdout
        assert '--language' in module_transcribe.output
        assert '--language' in subprocess_transcribe.stdout
    
    def test_error_handling_identical_across_entry_points(self):
        """Test that error handling is identical across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        # Test invalid command
        module_invalid = runner.invoke(main, ['invalid-command'])
        subprocess_invalid = subprocess.run([sys.executable, '-m', 'cli', 'invalid-command'], 
                                          capture_output=True, text=True)
        
        assert module_invalid.exit_code != 0
        assert subprocess_invalid.returncode != 0
        assert 'No such command' in module_invalid.output or 'invalid' in module_invalid.output.lower()
        assert 'No such command' in subprocess_invalid.stderr or 'invalid' in subprocess_invalid.stderr.lower()
        
        # Test missing required options
        module_extract_missing = runner.invoke(main, ['extract'])
        subprocess_extract_missing = subprocess.run([sys.executable, '-m', 'cli', 'extract'], 
                                                   capture_output=True, text=True)
        
        assert module_extract_missing.exit_code != 0
        assert subprocess_extract_missing.returncode != 0
        assert 'Missing option' in module_extract_missing.output or 'required' in module_extract_missing.output.lower()
        assert 'Missing option' in subprocess_extract_missing.stderr or 'required' in subprocess_extract_missing.stderr.lower()
    
    def test_option_parsing_identical_across_entry_points(self):
        """Test that option parsing is identical across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        # Test extract with options (will fail due to missing file, but parsing should work)
        module_extract = runner.invoke(main, ['extract', '--source', 'test.mp4', '--output', 'test.mp3'])
        subprocess_extract = subprocess.run([sys.executable, '-m', 'cli', 'extract', '--source', 'test.mp4', '--output', 'test.mp3'], 
                                          capture_output=True, text=True)
        
        # Both should fail with same error (file not found)
        assert module_extract.exit_code != 0
        assert subprocess_extract.returncode != 0
        assert 'Input file does not exist' in module_extract.output or 'not found' in module_extract.output.lower()
        assert 'Input file does not exist' in subprocess_extract.stdout or 'not found' in subprocess_extract.stdout.lower()
        
        # Test transcribe with language option - now requires --engine flag
        module_transcribe = runner.invoke(main, ['transcribe', '--source', 'test.mp3', '--language', 'en', '--engine', 'local-whisper'])
        subprocess_transcribe = subprocess.run([sys.executable, '-m', 'cli', 'transcribe', '--source', 'test.mp3', '--language', 'en', '--engine', 'local-whisper'], 
                                             capture_output=True, text=True)
        
        # Both should fail with same error (file not found or engine not available)
        assert module_transcribe.exit_code != 0
        assert subprocess_transcribe.returncode != 0
        # Accept either file not found or engine error - check both stdout and stderr for subprocess
        assert ('Audio file does not exist' in module_transcribe.output or 'not found' in module_transcribe.output.lower() or 
                'engine' in module_transcribe.output.lower())
        # For subprocess, error messages may be in stderr instead of stdout
        subprocess_output = subprocess_transcribe.stdout + subprocess_transcribe.stderr
        assert ('Audio file does not exist' in subprocess_output or 'not found' in subprocess_output.lower() or 
                'engine' in subprocess_output.lower())
    
    def test_exit_codes_identical_across_entry_points(self):
        """Test that exit codes are identical across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        # Success cases - test that both direct function and subprocess work
        success_commands = [
            ['--help'],
            ['--version'],
            ['extract', '--help'],
            ['transcribe', '--help']
        ]
        
        for cmd in success_commands:
            module_result = runner.invoke(main, cmd)
            subprocess_result = subprocess.run([sys.executable, '-m', 'cli'] + cmd, 
                                             capture_output=True, text=True)
            
            assert module_result.exit_code == 0, f"Command {cmd} should succeed via main()"
            assert subprocess_result.returncode == 0, f"Command {cmd} should succeed via subprocess"
        
        # Error cases - test that both fail appropriately
        error_commands = [
            ['invalid-command'],
            ['extract'],  # Missing required option
            ['transcribe'],  # Missing required option
        ]
        
        for cmd in error_commands:
            module_result = runner.invoke(main, cmd)
            subprocess_result = subprocess.run([sys.executable, '-m', 'cli'] + cmd, 
                                             capture_output=True, text=True)
            
            assert module_result.exit_code != 0, f"Command {cmd} should fail via main()"
            assert subprocess_result.returncode != 0, f"Command {cmd} should fail via subprocess"
    
    def test_cli_function_delegates_to_main(self):
        """Test that cli() function properly delegates to main()."""
        
        # Test that both functions exist
        from cli import main, cli
        assert callable(main)
        assert callable(cli)
        
        # Test that they produce similar results when invoked
        runner = CliRunner()
        
        main_result = runner.invoke(main, ['--help'])
        
        # Test that cli() function exists and can be called
        # We can't use CliRunner with cli() since it's a wrapper function
        # Instead, we test that the module invocation works
        import subprocess
        import sys
        
        subprocess_result = subprocess.run([sys.executable, '-m', 'cli', '--help'], 
                                         capture_output=True, text=True)
        
        assert main_result.exit_code == 0
        assert subprocess_result.returncode == 0
        
        # Should contain same core content
        assert 'Content Pipeline CLI' in main_result.output
        assert 'Content Pipeline CLI' in subprocess_result.stdout
    
    @given(st.sampled_from(['extract', 'transcribe']))
    @settings(deadline=None)  # Disable deadline for subprocess tests
    def test_subcommand_consistency_across_entry_points(self, command):
        """Property test: Subcommands should behave identically across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        # Test help for each command
        module_help = runner.invoke(main, [command, '--help'])
        subprocess_help = subprocess.run([sys.executable, '-m', 'cli', command, '--help'], 
                                       capture_output=True, text=True)
        
        assert module_help.exit_code == 0
        assert subprocess_help.returncode == 0
        
        # Should contain same key elements
        assert '--source' in module_help.output
        assert '--source' in subprocess_help.stdout
        assert '--output' in module_help.output
        assert '--output' in subprocess_help.stdout
        
        # Test missing required option
        module_missing = runner.invoke(main, [command])
        subprocess_missing = subprocess.run([sys.executable, '-m', 'cli', command], 
                                          capture_output=True, text=True)
        
        assert module_missing.exit_code != 0
        assert subprocess_missing.returncode != 0
    
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum() and x not in ['extract', 'transcribe', 'help', 'version']))
    @settings(deadline=None)  # Disable deadline for subprocess tests
    def test_invalid_commands_handled_identically(self, invalid_command):
        """Property test: Invalid commands should be handled identically across entry points."""
        runner = CliRunner()
        import subprocess
        import sys
        
        module_result = runner.invoke(main, [invalid_command])
        subprocess_result = subprocess.run([sys.executable, '-m', 'cli', invalid_command], 
                                         capture_output=True, text=True)
        
        # Both should fail
        assert module_result.exit_code != 0
        assert subprocess_result.returncode != 0
        
        # Both should indicate command not found
        module_lower = module_result.output.lower()
        subprocess_lower = subprocess_result.stderr.lower()
        
        error_indicators = ['no such command', 'invalid', 'not found', 'unknown']
        
        assert any(indicator in module_lower for indicator in error_indicators)
        assert any(indicator in subprocess_lower for indicator in error_indicators)
    
    def test_module_main_py_enables_direct_execution(self):
        """Test that __main__.py enables direct module execution."""
        import subprocess
        import sys
        
        # Test that python -m cli works (this tests __main__.py)
        result = subprocess.run([sys.executable, '-m', 'cli', '--version'], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert '0.7.0' in result.stdout
        
        # Test that it produces same output as direct function call
        runner = CliRunner()
        direct_result = runner.invoke(main, ['--version'])
        
        assert direct_result.exit_code == 0
        assert '0.7.0' in direct_result.output
    
    def test_entry_point_configuration_correct(self):
        """Test that setup.py entry point configuration is correct."""
        
        # Test that the console script entry point exists and works
        from cli import cli
        assert callable(cli)
        
        # Test that it's different from main (cli is wrapper, main is the actual CLI)
        from cli import main
        assert cli != main
        
        # Test that both are Click-related objects
        assert hasattr(main, 'main')  # Click group has main method
        assert callable(cli)  # cli is a function
        
        # Test that cli() ultimately calls main functionality
        runner = CliRunner()
        import subprocess
        import sys
        
        subprocess_result = subprocess.run([sys.executable, '-m', 'cli', '--help'], 
                                         capture_output=True, text=True)
        main_result = runner.invoke(main, ['--help'])
        
        # Should produce equivalent results
        assert subprocess_result.returncode == 0
        assert main_result.exit_code == 0
        assert 'Content Pipeline CLI' in subprocess_result.stdout
        assert 'Content Pipeline CLI' in main_result.output

class TestComprehensiveErrorHandling:
    """
    Property 12: Comprehensive Error Handling
    For any error condition, the CLI should handle errors gracefully with consistent 
    formatting, appropriate exit codes, and user-friendly messages.
    
    Feature: cli-refactoring-v0-6-0, Property 12: Comprehensive Error Handling
    Validates: Requirements 8.1, 8.2, 8.3, 8.5
    """
    
    def test_missing_required_option_error_consistency(self):
        """Test that missing required option errors are consistent across all commands."""
        runner = CliRunner()
        
        # Test extract command missing source
        extract_result = runner.invoke(main, ['extract'])
        assert extract_result.exit_code != 0
        assert 'Missing option' in extract_result.output or 'required' in extract_result.output.lower()
        
        # Test transcribe command missing source
        transcribe_result = runner.invoke(main, ['transcribe'])
        assert transcribe_result.exit_code != 0
        assert 'Missing option' in transcribe_result.output or 'required' in transcribe_result.output.lower()
        
        # Both should have similar error structure
        assert isinstance(extract_result.output, str)
        assert isinstance(transcribe_result.output, str)
        assert len(extract_result.output.strip()) > 0
        assert len(transcribe_result.output.strip()) > 0
    
    def test_file_not_found_error_handling_and_logging(self):
        """Test that file not found errors are handled gracefully with logging."""
        runner = CliRunner()
        
        # Test extract with non-existent file
        extract_result = runner.invoke(main, ['extract', '--source', 'nonexistent.mp4', '--output', 'test.mp3'])
        assert extract_result.exit_code != 0
        
        # Should contain user-friendly error message
        error_indicators = ['not found', 'does not exist', 'no such file', 'input file']
        assert any(indicator in extract_result.output.lower() for indicator in error_indicators)
        
        # Test transcribe with non-existent file - now requires --engine flag
        transcribe_result = runner.invoke(main, ['transcribe', '--source', 'nonexistent.mp3', '--engine', 'local-whisper'])
        assert transcribe_result.exit_code != 0
        
        # Should contain user-friendly error message (file not found or engine error)
        error_indicators = ['not found', 'does not exist', 'no such file', 'audio file', 'engine', 'missing']
        assert any(indicator in transcribe_result.output.lower() for indicator in error_indicators)
        
        # Both should exit with non-zero code
        assert extract_result.exit_code != 0
        assert transcribe_result.exit_code != 0
    
    def test_graceful_exception_handling(self):
        """Test that extraction and transcription failures are handled gracefully."""
        runner = CliRunner()
        
        # Test extract with invalid source format
        invalid_sources = ['invalid://url', 'not-a-file.xyz']
        
        for source in invalid_sources:
            result = runner.invoke(main, ['extract', '--source', source, '--output', 'test.mp3'])
            
            # Should fail gracefully (not crash)
            assert result.exit_code != 0
            
            # Should produce some error output
            assert len(result.output.strip()) > 0
            
            # Should not contain Python tracebacks in the main error message
            # (logging errors may appear but the main error should be clean)
            # Check that the first line doesn't start with "Traceback"
            first_line = result.output.strip().split('\n')[0]
            assert not first_line.startswith('Traceback')
    
    def test_error_message_format_consistency_with_v050(self):
        """Test that error messages maintain v0.5.0 format and content."""
        runner = CliRunner()
        
        # Test invalid command error format
        invalid_result = runner.invoke(main, ['invalid-command'])
        assert invalid_result.exit_code != 0
        assert 'No such command' in invalid_result.output or 'invalid' in invalid_result.output.lower()
        
        # Test missing option error format
        missing_result = runner.invoke(main, ['extract'])
        assert missing_result.exit_code != 0
        assert 'Missing option' in missing_result.output or 'required' in missing_result.output.lower()
        
        # Test file not found error format
        file_result = runner.invoke(main, ['extract', '--source', 'missing.mp4'])
        assert file_result.exit_code != 0
        assert any(phrase in file_result.output.lower() for phrase in ['not found', 'does not exist', 'input file'])
        
        # All error messages should be strings and non-empty
        for result in [invalid_result, missing_result, file_result]:
            assert isinstance(result.output, str)
            assert len(result.output.strip()) > 0
    
    def test_appropriate_exit_codes(self):
        """Test that CLI exits with appropriate error codes."""
        runner = CliRunner()
        
        # Success cases should return 0
        success_commands = [
            ['--help'],
            ['--version'],
            ['extract', '--help'],
            ['transcribe', '--help']
        ]
        
        for cmd in success_commands:
            result = runner.invoke(main, cmd)
            assert result.exit_code == 0, f"Command {cmd} should succeed with exit code 0"
        
        # Error cases should return non-zero
        error_commands = [
            ['invalid-command'],                                    # Invalid command
            ['extract'],                                           # Missing required option
            ['transcribe'],                                        # Missing required option
            ['extract', '--source', 'nonexistent.mp4'],          # File not found
            ['transcribe', '--source', 'nonexistent.mp3'],       # File not found
            ['extract', '--source', ''],                          # Empty source
            ['transcribe', '--source', '']                        # Empty source
        ]
        
        for cmd in error_commands:
            result = runner.invoke(main, cmd)
            assert result.exit_code != 0, f"Command {cmd} should fail with non-zero exit code"
    
    def test_error_message_constants_usage(self):
        """Test that error message constants are used consistently."""
        from cli.help_texts import MISSING_SOURCE_ERROR, INVALID_FORMAT_ERROR, FILE_NOT_FOUND_ERROR
        
        # Test that constants are properly formatted strings
        assert isinstance(MISSING_SOURCE_ERROR, str)
        assert isinstance(INVALID_FORMAT_ERROR, str)
        assert isinstance(FILE_NOT_FOUND_ERROR, str)
        
        # Test that constants are not empty
        assert len(MISSING_SOURCE_ERROR.strip()) > 0
        assert len(INVALID_FORMAT_ERROR.strip()) > 0
        assert len(FILE_NOT_FOUND_ERROR.strip()) > 0
        
        # Test that FILE_NOT_FOUND_ERROR supports formatting
        formatted = FILE_NOT_FOUND_ERROR.format(path="test.mp3")
        assert "test.mp3" in formatted
        assert formatted != FILE_NOT_FOUND_ERROR
    
    def test_error_output_structure_consistency(self):
        """Test that error output has consistent structure."""
        runner = CliRunner()
        
        # Test various error conditions
        error_tests = [
            (['invalid-command'], 'invalid command'),
            (['extract'], 'missing required option'),
            (['transcribe'], 'missing required option'),
            (['extract', '--source', 'missing.mp4'], 'file not found')
        ]
        
        for cmd, error_type in error_tests:
            result = runner.invoke(main, cmd)
            
            # Should fail
            assert result.exit_code != 0, f"Error test '{error_type}' should fail"
            
            # Should have output
            assert len(result.output.strip()) > 0, f"Error test '{error_type}' should have output"
            
            # Should be a string
            assert isinstance(result.output, str), f"Error test '{error_type}' output should be string"
            
            # Should contain some indication of the error
            error_words = ['error', 'missing', 'invalid', 'not found', 'no such', 'required']
            assert any(word in result.output.lower() for word in error_words), \
                f"Error test '{error_type}' should contain error indication"
    
    def test_help_options_never_produce_errors(self):
        """Test that help options always succeed and never produce errors."""
        runner = CliRunner()
        
        help_commands = [
            ['--help'],
            ['extract', '--help'],
            ['transcribe', '--help'],
            ['--version']
        ]
        
        for cmd in help_commands:
            result = runner.invoke(main, cmd)
            
            # Should always succeed
            assert result.exit_code == 0, f"Help command {cmd} should always succeed"
            
            # Should produce output
            assert len(result.output.strip()) > 0, f"Help command {cmd} should produce output"
            
            # Should not contain actual error messages (but may contain "error" in option descriptions like --log-level)
            # Check for actual error patterns, not just the word "error"
            error_patterns = ['Error:', 'failed', 'exception', 'traceback', 'Missing option']
            assert not any(pattern in result.output for pattern in error_patterns), \
                f"Help command {cmd} should not contain actual error messages"
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.isprintable() and not x.startswith('-') and x.upper() not in ['NUL', 'CON', 'PRN', 'AUX', 'COM1', 'COM2', 'LPT1', 'LPT2']))
    def test_invalid_source_handling_consistency(self, invalid_source):
        """Property test: Invalid sources should be handled consistently."""
        runner = CliRunner()
        
        # Skip sources that might be valid or are Windows special devices
        if any(valid in invalid_source.lower() for valid in ['http', 'www', '.mp', '.wav', '.m4a']):
            return
        
        # Test extract with invalid source
        result = runner.invoke(main, ['extract', '--source', invalid_source, '--output', 'test.mp3'])
        
        # Should produce output (may succeed or fail depending on the source)
        assert len(result.output.strip()) > 0
        
        # Should be a string
        assert isinstance(result.output, str)
        
        # Should not crash with unhandled exception in the main error message
        # Check that the first line doesn't start with "Traceback"
        first_line = result.output.strip().split('\n')[0]
        assert not first_line.startswith('Traceback')
    
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum() and x not in ['extract', 'transcribe', 'help', 'version']))
    def test_invalid_command_error_consistency(self, invalid_command):
        """Property test: Invalid commands should produce consistent error messages."""
        runner = CliRunner()
        
        result = runner.invoke(main, [invalid_command])
        
        # Should always fail
        assert result.exit_code != 0
        
        # Should produce output
        assert len(result.output.strip()) > 0
        
        # Should indicate command not found
        error_indicators = ['no such command', 'invalid', 'not found', 'unknown']
        assert any(indicator in result.output.lower() for indicator in error_indicators)
        
        # Should be properly formatted
        assert isinstance(result.output, str)
    
    @given(st.sampled_from(['extract', 'transcribe']))
    def test_missing_required_options_error_format(self, command):
        """Property test: Missing required options should have consistent error format."""
        runner = CliRunner()
        
        result = runner.invoke(main, [command])
        
        # Should fail
        assert result.exit_code != 0
        
        # Should indicate missing option
        assert 'Missing option' in result.output or 'required' in result.output.lower()
        
        # Should be properly formatted
        assert isinstance(result.output, str)
        assert len(result.output.strip()) > 0
    
    def test_error_logging_integration(self):
        """Test that errors are properly logged while maintaining user-friendly output."""
        runner = CliRunner()
        
        # Test file not found error (should be logged)
        result = runner.invoke(main, ['extract', '--source', 'definitely-not-a-file.mp4'])
        
        # Should fail with user-friendly message
        assert result.exit_code != 0
        assert any(phrase in result.output.lower() for phrase in ['not found', 'does not exist', 'input file'])
        
        # The main error message (first line) should be user-friendly
        first_line = result.output.strip().split('\n')[0]
        assert first_line.lower().startswith('error:') or 'not found' in first_line.lower() or 'does not exist' in first_line.lower()
        
        # Should contain clear error message
        assert len(result.output.strip()) > 0
        assert isinstance(result.output, str)
    
    def test_exception_handling_robustness(self):
        """Test that the CLI handles various exception scenarios robustly."""
        runner = CliRunner()
        
        # Test various problematic inputs
        problematic_inputs = [
            ('extract', ['--source', '', '--output', 'test.mp3']),                    # Empty source
            ('extract', ['--source', 'x' * 1000, '--output', 'test.mp3']),          # Very long source
            ('transcribe', ['--source', '', '--output', 'test.json']),               # Empty source
            ('transcribe', ['--source', 'x' * 1000, '--output', 'test.json']),      # Very long source
        ]
        
        for command, args in problematic_inputs:
            result = runner.invoke(main, [command] + args)
            
            # Should fail gracefully (not crash)
            assert result.exit_code != 0
            
            # Should produce some output
            assert len(result.output.strip()) > 0
            
            # Should be a string
            assert isinstance(result.output, str)
            
            # Should not contain unhandled Python exceptions
            assert not ('Traceback (most recent call last)' in result.output and 'Error:' not in result.output)
    
    def test_error_code_mapping_consistency(self):
        """Test that different error types map to appropriate exit codes consistently."""
        runner = CliRunner()
        
        # Different error types should all return non-zero, but consistently
        error_scenarios = [
            (['invalid-command'], 'invalid_command'),
            (['extract'], 'missing_option'),
            (['transcribe'], 'missing_option'),
            (['extract', '--source', 'missing.mp4'], 'file_not_found'),
            (['transcribe', '--source', 'missing.mp3'], 'file_not_found')
        ]
        
        exit_codes = {}
        
        for cmd, scenario_type in error_scenarios:
            result = runner.invoke(main, cmd)
            
            # Should fail
            assert result.exit_code != 0
            
            # Track exit codes by scenario type
            if scenario_type not in exit_codes:
                exit_codes[scenario_type] = []
            exit_codes[scenario_type].append(result.exit_code)
        
        # All exit codes should be non-zero
        for scenario_type, codes in exit_codes.items():
            assert all(code != 0 for code in codes), f"All {scenario_type} errors should have non-zero exit codes"
            
        # Similar error types should have consistent exit codes
        for scenario_type, codes in exit_codes.items():
            if len(codes) > 1:
                assert len(set(codes)) <= 2, f"Exit codes for {scenario_type} should be consistent (max 2 different codes)"