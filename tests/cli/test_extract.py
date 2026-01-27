"""
Unit Tests for Extract Subcommand

Test suite for the 'extract' subcommand of the content-pipeline CLI.
Covers extraction from different source types, option validation, and error handling.
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import pytest
from click.testing import CliRunner
from cli.extract import extract

CLI_PATH = "cli"  # Use module path for new CLI structure
TEST_OUTPUT_DIR = "tests/output"


class TestExtractCommand:
    """Unit tests for the extract command functionality."""
    
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
    
    def test_extract_help_output(self):
        """Test that extract command shows proper help output."""
        runner = CliRunner()
        result = runner.invoke(extract, ['--help'])
        
        assert result.exit_code == 0
        assert "Extract audio from the source file" in result.output
        assert "--source" in result.output
        assert "--output" in result.output
        assert "Streaming platform URL" in result.output
    
    def test_extract_missing_source_argument(self):
        """Test that extract command fails when source argument is missing."""
        runner = CliRunner()
        result = runner.invoke(extract, ['--output', 'test.mp3'])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    def test_extract_with_default_output(self):
        """Test that extract command uses default output when not specified."""
        runner = CliRunner()
        
        # Mock a file that doesn't exist to test the error path
        result = runner.invoke(extract, ['--source', 'nonexistent.mp4'])
        
        # Should fail because file doesn't exist, but should show it tried to use default output
        assert result.exit_code != 0
        # The command should have attempted to process with default output
    
    def test_extract_command_structure(self):
        """Test that extract command has proper Click structure."""
        # Test that the command is properly decorated
        assert hasattr(extract, 'callback')
        assert extract.name == 'extract'
        
        # Test that it has the expected parameters
        param_names = [param.name for param in extract.params]
        assert 'source' in param_names
        assert 'output' in param_names
    
    def test_extract_option_decorators(self):
        """Test that extract command uses shared option decorators correctly."""
        # Find the source option
        source_option = None
        output_option = None
        
        for param in extract.params:
            if param.name == 'source':
                source_option = param
            elif param.name == 'output':
                output_option = param
        
        # Test source option properties
        assert source_option is not None
        assert source_option.required is True
        assert '--source' in source_option.opts
        assert '-s' in source_option.opts
        
        # Test output option properties
        assert output_option is not None
        assert output_option.default == "output.mp3"
        assert '--output' in output_option.opts
        assert '-o' in output_option.opts


@pytest.mark.integration
class TestExtractIntegration:
    """Integration tests for extract command using subprocess calls."""
    
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
    
    def test_cli_extract_help_output(self):
        """Test extract command help output via CLI."""
        result = subprocess.run([sys.executable, "-m", "cli", "extract", "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Base filename for extracted audio (.mp3)" in result.stdout
        assert "--source" in result.stdout

    def test_cli_missing_arguments(self):
        """Verifies that the CLI exits with an error when required arguments are missing."""
        result = subprocess.run([
            sys.executable, "-m", "cli",         
            "extract"
            # No --source or --output provided
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert any(kw in result.stderr for kw in ["Missing argument", "Error:", "Missing option"])

    def test_cli_extract_missing_input_file(self, tmp_path):
        """Test extract command with non-existent input file."""
        missing_path = tmp_path / "nonexistent.mp4"
        output_path = tmp_path / "out.mp3"

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "extract",
            "--source", str(missing_path), 
            "--output", str(output_path)
        ], capture_output=True, text=True)

        assert result.returncode != 0
        assert "Input file does not exist" in result.stdout or "Input file does not exist" in result.stderr

    @pytest.mark.integration
    def test_cli_extract_youtube_audio_and_metadata_extraction(self, tmp_path):
        """
        Verifies that the CLI downloads audio from a YouTube URL and saves both MP3 and metadata files.
        """
        output_path = tmp_path / "cli_download.mp3"
        metadata_path = output_path.with_suffix(".json")
        url = "https://www.youtube.com/watch?v=qcOiqtMsjes"

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "extract",
            "--source", url,
            "--output", str(output_path)
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Validate metadata file
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)

            assert metadata["source_type"] == "streaming"
            assert metadata["source_url"] == url
            assert metadata["source_path"] is None
            assert metadata["metadata_status"] == "complete"
            assert isinstance(metadata["title"], str)
            assert isinstance(metadata["duration"], int)
            assert isinstance(metadata["author"], str)
            assert "service_metadata" in metadata
            assert isinstance(metadata["service_metadata"], dict)
            assert metadata["service_metadata"] != {}

        expected_keys = {
            "title", "duration", "author",
            "source_type", "source_path", "source_url",
            "metadata_status", "service_metadata"
        }
        assert expected_keys.issubset(metadata.keys())

        for path in [output_path, metadata_path]:
            if path.exists():
                path.unlink()

    @pytest.mark.integration
    def test_cli_extract_local_audio_and_placeholder_metadata(self, tmp_path):
        """
        Verifies that the CLI extracts audio from a local .mp4 file and generates placeholder metadata.
        """
        video_path = "tests/assets/sample_video.mp4"
        output_path = tmp_path / "cli_extracted.mp3"
        metadata_path = output_path.with_suffix(".json")

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "extract",
            "--source", video_path,
            "--output", str(output_path)
        ], capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Validate placeholder metadata
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
            assert metadata["title"] == os.path.basename(video_path)
            assert metadata["source_type"] == "file_system"
            assert metadata["source_path"] == os.path.abspath(video_path)
            assert metadata["source_url"] is None
            assert metadata["metadata_status"] == "incomplete"
            assert "service_metadata" in metadata
            assert isinstance(metadata["service_metadata"], dict)
            assert metadata["service_metadata"] == {}

        expected_keys = {
            "title", "duration", "author",
            "source_type", "source_path", "source_url",
            "metadata_status", "service_metadata"
        }
        assert expected_keys.issubset(metadata.keys())

        for path in [output_path, metadata_path]:
            if path.exists():
                path.unlink()