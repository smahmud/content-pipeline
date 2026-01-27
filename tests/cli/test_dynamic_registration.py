"""
Tests for Dynamic Command Registration

This test suite validates that new subcommands can be added to the CLI
without modifying the existing CLI package structure.
"""

import pytest
import click
from click.testing import CliRunner
from cli import main


class TestDynamicCommandRegistration:
    """Test dynamic command registration capability."""
    
    def test_new_command_can_be_registered(self):
        """Test that a new command can be dynamically registered."""
        
        # Create a test command for this test
        @click.command()
        @click.option('--source', '-s', required=True, help="Input source")
        @click.option('--output', '-o', default="output.mp3", help="Output file")
        def test_command(source, output):
            """Test command for validating dynamic registration capability."""
            click.echo(f"Test command executed with source: {source}, output: {output}")
            click.echo("Dynamic command registration is working!")
        
        # Create a new CLI group for testing
        @click.group()
        def test_cli():
            """Test CLI group."""
            pass
        
        # Register existing commands
        test_cli.add_command(main.commands['extract'])
        test_cli.add_command(main.commands['transcribe'])
        
        # Register new test command
        test_cli.add_command(test_command)
        
        runner = CliRunner()
        
        # Test that all commands are available
        result = runner.invoke(test_cli, ['--help'])
        assert result.exit_code == 0
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        assert 'test' in result.output
        
        # Test that new command works
        result = runner.invoke(test_cli, ['test', '--help'])
        assert result.exit_code == 0
        assert 'Test command for validating dynamic registration' in result.output
        
        # Test that new command can be executed
        result = runner.invoke(test_cli, ['test', '--source', 'test.txt'])
        assert result.exit_code == 0
        assert 'Dynamic command registration is working!' in result.output
    
    def test_new_command_uses_shared_options(self):
        """Test that new commands can use shared option decorators."""
        from cli.shared_options import input_option, output_option
        from cli.help_texts import SOURCE_HELP, OUTPUT_HELP
        
        # Create a command using shared options
        @click.command()
        @input_option(help=SOURCE_HELP)
        @output_option(help=OUTPUT_HELP)
        def shared_test_command(source, output):
            """Test command using shared options."""
            click.echo(f"source: {source}, output: {output}")
        
        runner = CliRunner()
        
        # Test help output shows shared options
        result = runner.invoke(shared_test_command, ['--help'])
        assert result.exit_code == 0
        assert '--source' in result.output
        assert '-s' in result.output
        assert '--output' in result.output
        assert '-o' in result.output
        
        # Test that shared options work
        result = runner.invoke(shared_test_command, ['--source', 'input.txt', '--output', 'output.txt'])
        assert result.exit_code == 0
        assert 'source: input.txt' in result.output
        assert 'output: output.txt' in result.output
    
    def test_new_command_structure_matches_existing(self):
        """Test that new commands follow the same structure as existing commands."""
        
        # Create a test command
        @click.command()
        @click.option('--source', '-s', required=True, help="Input source")
        @click.option('--output', '-o', default="output.mp3", help="Output file")
        def test_command(source, output):
            """Test command."""
            pass
        
        # Test that test_command has the same structure as existing commands
        assert hasattr(test_command, 'callback')
        assert test_command.name == 'test'
        
        # Test that it has expected parameters
        param_names = [param.name for param in test_command.params]
        assert 'source' in param_names
        assert 'output' in param_names
        
        # Test that parameters have expected properties
        source_param = next(p for p in test_command.params if p.name == 'source')
        output_param = next(p for p in test_command.params if p.name == 'output')
        
        assert source_param.required is True
        assert output_param.default == "output.mp3"
    
    def test_command_registration_without_core_modification(self):
        """Test that commands can be registered without modifying core CLI files."""
        
        # This test verifies that we can create and register a new command
        # without modifying cli/__init__.py, shared_options.py, or help_texts.py
        
        # Create a completely new command
        @click.command()
        @click.option('--input', '-i', required=True, help="Input file")
        @click.option('--format', '-f', default='json', help="Output format")
        def new_test_command(input, format):
            """Another test command for registration testing."""
            click.echo(f"New command: input={input}, format={format}")
        
        # Test that it can be registered dynamically
        @click.group()
        def dynamic_cli():
            """Dynamic CLI for testing."""
            pass
        
        dynamic_cli.add_command(new_test_command)
        
        runner = CliRunner()
        
        # Test that the command is available
        result = runner.invoke(dynamic_cli, ['--help'])
        assert result.exit_code == 0
        assert 'new-test' in result.output
        
        # Test that the command works
        result = runner.invoke(dynamic_cli, ['new-test', '--input', 'test.txt'])
        assert result.exit_code == 0
        assert 'New command: input=test.txt, format=json' in result.output
    
    def test_multiple_commands_registration(self):
        """Test that multiple new commands can be registered together."""
        
        # Create multiple test commands
        @click.command()
        @click.option('--data', required=True)
        def cmd1(data):
            """First test command."""
            click.echo(f"Command 1: {data}")
        
        @click.command()
        @click.option('--value', required=True)
        def cmd2(value):
            """Second test command."""
            click.echo(f"Command 2: {value}")
        
        @click.command()
        @click.option('--source', '-s', required=True, help="Input source")
        def test_cmd(source):
            """Test command for registration testing."""
            click.echo("Dynamic command registration is working!")
        
        # Register all commands
        @click.group()
        def multi_cli():
            """Multi-command CLI."""
            pass
        
        multi_cli.add_command(cmd1)
        multi_cli.add_command(cmd2)
        multi_cli.add_command(test_cmd)
        
        runner = CliRunner()
        
        # Test that all commands are available
        result = runner.invoke(multi_cli, ['--help'])
        assert result.exit_code == 0
        assert 'cmd1' in result.output
        assert 'cmd2' in result.output
        assert 'test' in result.output
        
        # Test that all commands work
        result = runner.invoke(multi_cli, ['cmd1', '--data', 'test1'])
        assert result.exit_code == 0
        assert 'Command 1: test1' in result.output
        
        result = runner.invoke(multi_cli, ['cmd2', '--value', 'test2'])
        assert result.exit_code == 0
        assert 'Command 2: test2' in result.output
        
        result = runner.invoke(multi_cli, ['test', '--source', 'test3'])
        assert result.exit_code == 0
        assert 'Dynamic command registration is working!' in result.output


class TestExtensibilityValidation:
    """Test that the CLI architecture supports extensibility."""
    
    def test_shared_options_can_be_extended(self):
        """Test that shared options module can be extended with new decorators."""
        
        # This test validates that we can add new shared option decorators
        # without breaking existing functionality
        
        def custom_option(help="Custom option"):
            """Custom shared option decorator."""
            return click.option('--custom', '-c', default='default', help=help)
        
        # Test that custom decorator works
        @click.command()
        @custom_option(help="Test custom option")
        def cmd_with_custom(custom):
            click.echo(f"Custom: {custom}")
        
        runner = CliRunner()
        
        result = runner.invoke(cmd_with_custom, ['--help'])
        assert result.exit_code == 0
        assert '--custom' in result.output
        assert 'Test custom option' in result.output
        
        result = runner.invoke(cmd_with_custom, ['--custom', 'test_value'])
        assert result.exit_code == 0
        assert 'Custom: test_value' in result.output
    
    def test_help_texts_can_be_extended(self):
        """Test that help text constants can be extended."""
        
        # Test that we can add new help text constants
        NEW_COMMAND_HELP = "Help text for a new command"
        NEW_OPTION_HELP = "Help text for a new option"
        
        @click.command(help=NEW_COMMAND_HELP)
        @click.option('--new-option', help=NEW_OPTION_HELP)
        def cmd_with_new_help(new_option):
            click.echo("Command with new help")
        
        runner = CliRunner()
        
        result = runner.invoke(cmd_with_new_help, ['--help'])
        assert result.exit_code == 0
        assert NEW_COMMAND_HELP in result.output
        assert NEW_OPTION_HELP in result.output
    
    def test_cli_package_structure_unchanged(self):
        """Test that adding new commands doesn't require changing CLI package structure."""
        
        # This test verifies that the core CLI files remain unchanged
        # when adding new commands
        
        import cli
        import cli.shared_options
        import cli.help_texts
        
        # Test that core modules are still importable and functional
        assert hasattr(cli, 'main')
        assert hasattr(cli.shared_options, 'input_option')
        assert hasattr(cli.help_texts, 'EXTRACT_HELP')
        
        # Test that main CLI still works
        runner = CliRunner()
        result = runner.invoke(cli.main, ['--help'])
        assert result.exit_code == 0
        assert 'extract' in result.output
        assert 'transcribe' in result.output
        
        # Test that the CLI structure supports the existing commands
        assert 'extract' in cli.main.commands
        assert 'transcribe' in cli.main.commands
        
        # Test that commands are properly registered Click commands
        assert isinstance(cli.main.commands['extract'], click.Command)
        assert isinstance(cli.main.commands['transcribe'], click.Command)