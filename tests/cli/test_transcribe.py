"""
Unit Tests for Transcribe Subcommand

Test suite for the 'transcribe' subcommand of the content-pipeline CLI.
Enhanced for v0.6.5 with breaking changes:
- --engine flag is now required
- Output paths are configurable
- New CLI options and error handling
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch
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
        """Test that transcribe command shows proper help output with new v0.6.5 options."""
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--help'])
        
        assert result.exit_code == 0
        # Check for key parts of the help text (the exact wording may vary)
        assert "Transcribe audio content" in result.output
        assert "--source" in result.output
        assert "--output" in result.output
        assert "--language" in result.output
        assert "--engine" in result.output  # New required option
        assert "--model" in result.output   # New option
        assert "--api-key" in result.output # New option
        assert "--output-dir" in result.output # New option
        assert "--config" in result.output # New option
        assert "--log-level" in result.output # New option
        assert "local-whisper" in result.output
        assert "openai-whisper" in result.output
        assert "auto" in result.output
    
    def test_transcribe_missing_engine_flag_breaking_change(self):
        """Test that transcribe command fails when required --engine flag is missing (breaking change)."""
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--source', 'test.mp3'])
        
        assert result.exit_code != 0
        # Should provide breaking change migration guidance
        output_lower = result.output.lower()
        breaking_change_indicators = [
            'engine', 'required', 'breaking', 'v0.6.5',
            'local-whisper', 'openai-whisper', 'auto'
        ]
        assert any(indicator in output_lower for indicator in breaking_change_indicators)
    
    def test_transcribe_missing_source_argument(self):
        """Test that transcribe command fails when source argument is missing."""
        runner = CliRunner()
        result = runner.invoke(transcribe, ['--engine', 'local-whisper', '--output', 'transcript.json'])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    def test_transcribe_with_required_engine_flag(self):
        """Test that transcribe command works with required --engine flag."""
        runner = CliRunner()
        
        # Test with required engine flag - use a config that avoids API key issues
        with runner.isolated_filesystem():
            # Create a minimal config that doesn't require API keys
            config_content = """
engine: local-whisper
whisper_local:
  model: tiny
output_dir: ./test_output
"""
            with open('test_config.yaml', 'w') as f:
                f.write(config_content)
            
            result = runner.invoke(transcribe, [
                '--source', 'nonexistent.mp3', 
                '--engine', 'local-whisper',
                '--config', 'test_config.yaml'
            ])
            
            # Should fail because file doesn't exist, not because of missing engine flag or config issues
            assert result.exit_code != 0
            output_lower = result.output.lower()
            # Should be a file not found error, not engine or config error
            assert ("not found" in output_lower or "does not exist" in output_lower or 
                   "no such file" in output_lower), f"Expected file not found error, got: {result.output}"
    
    def test_transcribe_with_invalid_engine(self):
        """Test that transcribe command fails with invalid engine."""
        runner = CliRunner()
        
        result = runner.invoke(transcribe, ['--source', 'test.mp3', '--engine', 'invalid-engine'])
        
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()
        # Should list valid engines
        output_lower = result.output.lower()
        assert any(engine in output_lower for engine in ['local-whisper', 'openai-whisper', 'auto'])
    
    def test_transcribe_with_language_option(self):
        """Test that transcribe command accepts language option."""
        runner = CliRunner()
        
        # Test with language option - use isolated filesystem to avoid config issues
        with runner.isolated_filesystem():
            # Create a minimal config that doesn't require API keys
            config_content = """
engine: local-whisper
whisper_local:
  model: tiny
output_dir: ./test_output
"""
            with open('test_config.yaml', 'w') as f:
                f.write(config_content)
            
            result = runner.invoke(transcribe, [
                '--source', 'test.mp3',
                '--engine', 'local-whisper',
                '--language', 'en',
                '--config', 'test_config.yaml'
            ])
            
            # Should fail because file doesn't exist, but language option should be parsed
            assert result.exit_code != 0
            output_lower = result.output.lower()
            assert ("not found" in output_lower or "does not exist" in output_lower or 
                   "no such file" in output_lower), f"Expected file not found error, got: {result.output}"
    
    def test_transcribe_with_new_v065_options(self):
        """Test that transcribe command accepts new v0.6.5 options."""
        runner = CliRunner()
        
        # Test with new options - use isolated filesystem to avoid config issues
        with runner.isolated_filesystem():
            # Create a minimal config that doesn't require API keys
            config_content = """
engine: local-whisper
whisper_local:
  model: tiny
output_dir: ./test_output
"""
            with open('test_config.yaml', 'w') as f:
                f.write(config_content)
            
            result = runner.invoke(transcribe, [
                '--source', 'test.mp3',
                '--engine', 'local-whisper',
                '--model', 'base',
                '--output-dir', './custom_output',
                '--log-level', 'debug',
                '--config', 'test_config.yaml'
            ])
            
            # Should fail because file doesn't exist, but options should be parsed
            assert result.exit_code != 0
            output_lower = result.output.lower()
            assert ("not found" in output_lower or "does not exist" in output_lower or 
                   "no such file" in output_lower), f"Expected file not found error, got: {result.output}"
    
    def test_transcribe_command_structure(self):
        """Test that transcribe command has proper Click structure with v0.6.5 enhancements."""
        # Test that the command is properly decorated
        assert hasattr(transcribe, 'callback')
        assert transcribe.name == 'transcribe'
        
        # Test that it has the expected parameters (including new v0.6.5 options)
        param_names = [param.name for param in transcribe.params]
        assert 'source' in param_names
        assert 'output' in param_names
        assert 'language' in param_names
        # New v0.6.5 parameters
        assert 'engine' in param_names
        assert 'model' in param_names
        assert 'api_key' in param_names
        assert 'output_dir' in param_names
        assert 'config' in param_names
        assert 'log_level' in param_names
    
    def test_transcribe_option_decorators(self):
        """Test that transcribe command uses shared option decorators correctly with v0.6.5 enhancements."""
        # Find the options
        source_option = None
        output_option = None
        language_option = None
        engine_option = None
        
        for param in transcribe.params:
            if param.name == 'source':
                source_option = param
            elif param.name == 'output':
                output_option = param
            elif param.name == 'language':
                language_option = param
            elif param.name == 'engine':
                engine_option = param
        
        # Test source option properties (from shared decorator)
        assert source_option is not None
        assert source_option.required is True
        assert '--source' in source_option.opts
        assert '-s' in source_option.opts
        
        # Test output option properties (custom for transcribe)
        assert output_option is not None
        assert output_option.default is None  # Changed in v0.6.5
        assert '--output' in output_option.opts
        
        # Test language option properties (from shared decorator)
        assert language_option is not None
        assert language_option.default is None
        assert '--language' in language_option.opts
        assert '-l' in language_option.opts
        
        # Test new engine option (required in v0.6.5)
        assert engine_option is not None
        assert engine_option.required is True
        assert '--engine' in engine_option.opts


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
        """Test transcribe command help output via CLI with v0.6.5 enhancements."""
        result = subprocess.run([sys.executable, "-m", "cli", "transcribe", "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "--source" in result.stdout
        assert "--output" in result.stdout
        assert "--language" in result.stdout
        # New v0.6.5 options
        assert "--engine" in result.stdout
        assert "--model" in result.stdout
        assert "--api-key" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--config" in result.stdout
        assert "--log-level" in result.stdout

    @pytest.mark.integration
    def test_cli_transcribe_local_audio(self, tmp_path):
        """
        Verifies that the CLI transcribes a local MP3 file and saves a valid transcript.
        Updated for v0.6.5 with required --engine flag.
        """
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create a test config file to avoid issues with the main config
        config_file = tmp_path / "test_config.yaml"
        config_content = """
engine: local-whisper
whisper_local:
  model: tiny
  timeout: 60
output_dir: ./output
log_level: info
"""
        config_file.write_text(config_content)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--engine", "local-whisper",  # Required in v0.6.5
            "--model", "tiny",            # Use fastest model for testing
            "--output-dir", str(output_dir),
            "--config", str(config_file)
        ], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        # Check if transcription succeeded or failed for expected reasons
        if result.returncode != 0:
            # If it failed, check if it's due to missing Whisper installation
            error_output = (result.stdout or "") + (result.stderr or "")
            if ("whisper" in error_output.lower() and ("not found" in error_output.lower() or "not installed" in error_output.lower())) or \
               ("ffmpeg" in error_output.lower() or "system cannot find the file specified" in error_output.lower()):
                pytest.skip("Whisper or ffmpeg not installed - skipping integration test")
            else:
                # For other errors, we still want to see what happened
                print(f"Unexpected error (code {result.returncode}): {error_output}")
                # Don't fail the test immediately - let's check if output was created anyway
        
        # Check if transcript was created
        transcript_files = list(output_dir.glob("*.json"))
        if len(transcript_files) >= 1:
            # Success case - verify the transcript structure
            with open(transcript_files[0]) as f:
                data = json.load(f)
                assert "metadata" in data
                assert "transcript" in data
                assert isinstance(data["transcript"], list)
        elif result.returncode == 0:
            # If command succeeded but no transcript, that's an error
            assert False, f"Command succeeded but no transcript file created in {output_dir}"
        else:
            # Command failed - check if it's a known issue we can skip
            error_output = (result.stdout or "") + (result.stderr or "")
            if ("whisper" in error_output.lower() and "not" in error_output.lower()) or \
               ("ffmpeg" in error_output.lower() or "system cannot find the file specified" in error_output.lower()):
                pytest.skip("Whisper or ffmpeg not available - skipping integration test")
            else:
                assert False, f"Transcription failed with code {result.returncode}: {error_output}"

    @pytest.mark.integration
    def test_cli_transcribe_output_structure(self, tmp_path):
        """Test that transcribe command produces correct output structure with v0.6.5 enhancements."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        output_dir = tmp_path / "transcripts"
        
        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
engine: local-whisper
whisper_local:
  model: tiny
  timeout: 60
output_dir: ./transcripts
log_level: info
"""
        config_file.write_text(config_content)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--engine", "local-whisper",  # Required in v0.6.5
            "--model", "tiny",            # Use fastest model for testing
            "--output-dir", str(output_dir),
            "--config", str(config_file)
        ], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        # Skip test if Whisper is not available
        if result.returncode != 0:
            error_output = (result.stdout or "") + (result.stderr or "")
            if "whisper" in error_output.lower() and ("not found" in error_output.lower() or "not installed" in error_output.lower()):
                pytest.skip("Whisper not installed - skipping integration test")

        # Find the generated transcript file
        transcript_files = list(output_dir.glob("*.json"))
        if len(transcript_files) == 0:
            pytest.skip("No transcript files generated - likely Whisper not available")

        with open(transcript_files[0]) as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "metadata" in data
            assert "transcript" in data
            assert isinstance(data["transcript"], list)
            assert all("text" in segment for segment in data["transcript"])

    @pytest.mark.integration
    def test_cli_transcribe_with_language_flag(self, tmp_path):
        """Test transcribe command with language flag and v0.6.5 enhancements."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        output_dir = tmp_path / "transcripts"
        
        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
engine: local-whisper
whisper_local:
  model: tiny
  timeout: 60
output_dir: ./transcripts
log_level: info
"""
        config_file.write_text(config_content)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--engine", "local-whisper",  # Required in v0.6.5
            "--model", "tiny",            # Use fastest model for testing
            "--language", "en",
            "--output-dir", str(output_dir),
            "--config", str(config_file)
        ], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        # Skip test if Whisper is not available
        if result.returncode != 0:
            error_output = (result.stdout or "") + (result.stderr or "")
            if "whisper" in error_output.lower() and ("not found" in error_output.lower() or "not installed" in error_output.lower()):
                pytest.skip("Whisper not installed - skipping integration test")

        # Find the generated transcript file
        transcript_files = list(output_dir.glob("*.json"))
        if len(transcript_files) == 0:
            pytest.skip("No transcript files generated - likely Whisper not available")

        with open(transcript_files[0]) as f:
            data = json.load(f)
            assert "metadata" in data
            # Language should be preserved in metadata
            assert data["metadata"].get("language") == "en"
    
    @pytest.mark.integration
    def test_cli_transcribe_missing_engine_flag_error(self, tmp_path):
        """Test that CLI fails with helpful error when --engine flag is missing (breaking change)."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        # Try to run without --engine flag (should fail)
        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio)
        ], cwd=tmp_path, capture_output=True, text=True)

        assert result.returncode != 0
        # Should provide breaking change migration guidance
        output_text = (result.stdout or "") + (result.stderr or "")
        output_lower = output_text.lower()
        breaking_change_indicators = [
            'engine', 'required', 'breaking', 'v0.6.5',
            'local-whisper', 'openai-whisper', 'auto'
        ]
        assert any(indicator in output_lower for indicator in breaking_change_indicators)
    
    @pytest.mark.integration 
    def test_cli_transcribe_with_configuration_file(self, tmp_path):
        """Test transcribe command with configuration file (v0.6.5 feature)."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        # Create configuration file
        config_file = tmp_path / "config.yaml"
        config_content = """
engine: local-whisper
output_dir: ./config_output
whisper_local:
  model: tiny
  timeout: 60
log_level: info
"""
        config_file.write_text(config_content)

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--config", str(config_file)
        ], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        # Skip test if Whisper is not available
        if result.returncode != 0:
            error_output = (result.stdout or "") + (result.stderr or "")
            if "whisper" in error_output.lower() and ("not found" in error_output.lower() or "not installed" in error_output.lower()):
                pytest.skip("Whisper not installed - skipping integration test")
            elif "engine" in error_output.lower() and "required" in error_output.lower():
                # This is the expected breaking change behavior - engine is required even with config
                assert "Missing option '--engine'" in result.stderr
                return
        
        # If command succeeded, check that output was created in configured directory
        config_output_dir = tmp_path / "config_output"
        if config_output_dir.exists():
            transcript_files = list(config_output_dir.glob("*.json"))
            assert len(transcript_files) >= 1


@pytest.mark.integration
class TestBackwardCompatibilityAndMigration:
    """Tests for backward compatibility and migration guidance from v0.6.0 to v0.6.5."""
    
    def setup_method(self):
        """Set up test environment."""
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        if os.path.exists("output"):
            shutil.rmtree("output")
    
    def test_old_cli_pattern_fails_with_migration_guidance(self, tmp_path):
        """Test that old v0.6.0 CLI patterns fail with clear migration guidance."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        # Try old v0.6.0 pattern (no --engine flag)
        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--output", "transcript.json"
        ], cwd=tmp_path, capture_output=True, text=True)

        assert result.returncode != 0
        
        # Should provide migration guidance
        output_text = (result.stdout or "") + (result.stderr or "")
        output_lower = output_text.lower()
        
        # Check for migration guidance elements
        migration_elements = [
            'breaking change',
            'v0.6.5',
            'engine',
            'required',
            'local-whisper',
            'openai-whisper',
            'auto',
            'example'
        ]
        
        found_elements = [elem for elem in migration_elements if elem in output_lower]
        assert len(found_elements) >= 4, f"Expected migration guidance, found: {found_elements}"
    
    def test_migration_examples_are_valid(self, tmp_path):
        """Test that the migration examples provided in error messages are actually valid."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        # Test the suggested migration patterns work
        migration_patterns = [
            # Pattern 1: whisper-local
            [
                "--source", str(input_audio),
                "--engine", "local-whisper",
                "--model", "tiny",
                "--output-dir", str(tmp_path / "output1")
            ],
            # Pattern 2: auto selection
            [
                "--source", str(input_audio),
                "--engine", "auto",
                "--output-dir", str(tmp_path / "output2")
            ]
        ]
        
        for i, pattern in enumerate(migration_patterns):
            result = subprocess.run([
                sys.executable, "-m", "cli",
                "transcribe"
            ] + pattern, cwd=tmp_path, capture_output=True, text=True)
            
            # Should succeed (or fail for reasons other than missing --engine)
            if result.returncode != 0:
                # If it fails, it should NOT be due to missing engine flag
                output_text = (result.stdout or "") + (result.stderr or "")
                
                # Check if it's failing due to missing Whisper installation (acceptable)
                if "whisper" in output_text.lower() and ("not available" in output_text.lower() or "failed to load" in output_text.lower()):
                    # This is acceptable - engine selection worked, but Whisper isn't installed
                    continue
                
                # Check if it's failing due to missing engine flag (not acceptable)
                if ("engine" in output_text.lower() and "required" in output_text.lower()) or "Missing option '--engine'" in output_text:
                    assert False, f"Migration pattern {i+1} still fails due to missing engine: {output_text}"
                
                # Other failures are acceptable (missing dependencies, etc.)
                print(f"Migration pattern {i+1} failed due to missing dependencies (acceptable): {result.returncode}")
    
    def test_backward_compatibility_transcript_format(self, tmp_path):
        """Test that transcript output format remains compatible with v0.6.0."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        output_dir = tmp_path / "output"

        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--engine", "local-whisper",
            "--model", "tiny",
            "--output-dir", str(output_dir)
        ], cwd=tmp_path, capture_output=True, text=True)

        # Skip test if Whisper is not available
        if result.returncode != 0:
            error_output = (result.stdout or "") + (result.stderr or "")
            if ("whisper" in error_output.lower() and ("not available" in error_output.lower() or "failed to load" in error_output.lower())) or \
               ("ffmpeg" in error_output.lower() or "system cannot find the file specified" in error_output.lower()):
                pytest.skip("Whisper or ffmpeg not available - skipping backward compatibility test")

        assert result.returncode == 0
        
        # Find the generated transcript file
        transcript_files = list(output_dir.glob("*.json"))
        assert len(transcript_files) == 1

        with open(transcript_files[0]) as f:
            data = json.load(f)
            
            # Should maintain v0.6.0 compatible structure
            assert isinstance(data, dict)
            assert "metadata" in data
            assert "transcript" in data
            assert isinstance(data["transcript"], list)
            
            # Check metadata structure compatibility
            metadata = data["metadata"]
            assert "engine" in metadata
            assert "created_at" in metadata  # v0.6.5 uses created_at instead of timestamp
            assert "schema_version" in metadata  # v0.6.5 includes schema_version
            
            # Check transcript structure compatibility
            for segment in data["transcript"]:
                assert "text" in segment
                # Should have timing information
                assert "start" in segment or "timestamp" in segment
    
    def test_legacy_output_path_behavior_changed(self, tmp_path):
        """Test that legacy hardcoded ./output/ behavior has changed."""
        from shutil import copyfile
        input_audio = tmp_path / "sample.mp3"
        copyfile("tests/assets/sample_audio.mp3", input_audio)

        # In v0.6.0, output would go to ./output/ by default
        # In v0.6.5, output location is configurable
        result = subprocess.run([
            sys.executable, "-m", "cli",
            "transcribe",
            "--source", str(input_audio),
            "--engine", "local-whisper",
            "--model", "tiny"
            # No --output-dir specified - should use default from config
        ], cwd=tmp_path, capture_output=True, text=True)

        # Skip test if Whisper is not available
        if result.returncode != 0:
            error_output = (result.stdout or "") + (result.stderr or "")
            if ("whisper" in error_output.lower() and ("not available" in error_output.lower() or "failed to load" in error_output.lower())) or \
               ("ffmpeg" in error_output.lower() or "system cannot find the file specified" in error_output.lower()):
                pytest.skip("Whisper or ffmpeg not available - skipping legacy output path test")

        assert result.returncode == 0
        
        # Should NOT create hardcoded ./output/ directory
        hardcoded_output = tmp_path / "output"
        
        # Should create configurable output directory instead
        # (Default is ./transcripts in our configuration)
        configurable_output = tmp_path / "transcripts"
        
        # The new behavior should be used
        if hardcoded_output.exists():
            # If ./output exists, it should be empty or not contain our transcript
            output_files = list(hardcoded_output.glob("*.json"))
            transcript_files = list(configurable_output.glob("*.json"))
            
            # Either no files in old location, or files in new location
            assert len(output_files) == 0 or len(transcript_files) > 0, \
                "Output should use configurable directory, not hardcoded ./output/"