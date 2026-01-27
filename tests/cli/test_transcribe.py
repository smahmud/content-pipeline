"""
Unit Tests for Transcribe Subcommand

Test suite for the 'transcribe' subcommand of the content-pipeline CLI.
Covers transcription with different options, language handling, and error conditions.
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import pytest
from click.testing import CliRunner
from cli.transcribe import transcribe

CLI_PATH = "cli"  # Use module path for new CLI structure
TEST_OUTPUT_DIR = "tests/output"


class TestTranscribeCommand:
    """Unit tests for the transcribe command functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clean up any existing test output
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Clean up test output
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def test_transcribe_help_output(self):
        """Test that transcribe command shows proper help output."""
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--help'])
        
        assert result.exit_code == 0
        assert "Extract audio from the source, run transcription" in result.output
        assert "--source" in result.output
        assert "--output" in result.output
        assert "--language" in result.output
        assert "Optional language hint" in result.output
    
    def test_transcribe_missing_source_argument(self):
        """Test that transcribe command fails when source argument is missing."""
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--output', 'transcript.json'])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    def test_transcribe_with_default_output(self):
        """Test that transcribe command uses default output when not specified."""
        runner = CliRunner()
        
        # Mock a file that doesn't exist to test the error path
        result = runner.invoke(transcribe, ['--source', 'nonexistent.mp3'])
        
        # Should fail because file doesn't exist
        assert result.exit_code != 0
        assert "Error: Audio file does not exist" in result.output
    
    def test_transcribe_with_language_option(self):
        """Test that transcribe command accepts language option."""
        runner = CliRunner()
        
        # Test with language option (will fail due to missing file, but should parse correctly)
        result = runner.invoke(transcribe, ['--source', 'test.mp3', '--language', 'en'])
        
        # Should fail because file doesn't exist, but language option should be parsed
        assert result.exit_code != 0
        assert "Error: Audio file does not exist" in result.output
    
    def test_transcribe_command_structure(self):
        """Test that transcribe command has proper Click structure."""
        # Test that the command is properly decorated
        assert hasattr(transcribe, 'callback')
        assert transcribe.name == 'transcribe'
        
        # Test that it has the expected parameters
        param_names = [param.name for param in transcribe.params]
        assert 'source' in param_names
        assert 'output' in param_names
        assert 'language' in param_names
    
    def test_transcribe_option_decorators(self):
        """Test that transcribe command uses shared option decorators correctly."""
        # Find the options
        source_option = None
        output_option = None
        language_option = None
        
        for param in transcribe.params:
            if param.name == 'source':
                source_option = param
            elif param.name == 'output':
                output_option = param
            elif param.name == 'language':
                language_option = param
        
        # Test source option properties (from shared decorator)
        assert source_option is not None
        assert source_option.required is True
        assert '--source' in source_option.opts
        assert '-s' in source_option.opts
        
        # Test output option properties (custom for transcribe)
        assert output_option is not None
        assert output_option.default == "transcript.json"
        assert '--output' in output_option.opts
        
        # Test language option properties (from shared decorator)
        assert language_option is not None
        assert language_option.default is None
        assert '--language' in language_option.opts
        assert '-l' in language_option.opts


@pytest.mark.integration
class TestTranscribeIntegration:
    """Integration tests for transcribe command using subprocess calls."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clean up any existing test output
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Clean up test output
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def test_cli_transcribe_help_output(self):
        """Test transcribe command help output via CLI."""
        result = subprocess.run([sys.executable, "-m", "cli", "transcribe", "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "--source" in result.stdout
        assert "--output" in result.stdout
        assert "--language" in result.stdout

    @pytest.mark.integration
    def test_cli_transcribe_local_audio(self, tmp_path):
        """
        Verifies that the CLI transcribes a local MP3 file and saves a valid transcript.
        """
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        output_path = tmp_path / "output" / "transcript.json"

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--output", str(output_path.name)
        ], cwd=tmp_path, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
            assert "metadata" in data
            assert "transcript" in data
            assert isinstance(data["transcript"], list)

    @pytest.mark.integration
    def test_cli_transcribe_output_structure(self, tmp_path):
        """Test that transcribe command produces correct output structure."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio)
        ], cwd=tmp_path, capture_output=True, text=True)

        output_path = tmp_path / "output" / "transcript.json"
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "metadata" in data
            assert "transcript" in data
            assert isinstance(data["transcript"], list)
            assert all("text" in segment for segment in data["transcript"])

    @pytest.mark.integration
    def test_cli_transcribe_with_language_flag(self, tmp_path):
        """Test transcribe command with language flag."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--language", "en"
        ], cwd=tmp_path, capture_output=True, text=True)

        output_path = tmp_path / "output" / "transcript.json"
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
            assert "metadata" in data
            assert data["metadata"].get("language") == "en"