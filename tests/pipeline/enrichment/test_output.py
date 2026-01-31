"""
Unit tests for Output File Management

Tests output path generation, validation, and file writing.
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, Mock

from pipeline.enrichment.output import (
    generate_output_filename,
    resolve_output_path,
    validate_output_path,
    check_overwrite,
    write_enrichment_result,
    write_batch_results,
    OutputManager
)
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata
from pipeline.enrichment.errors import OutputFileError
from datetime import datetime


@pytest.fixture
def sample_enrichment():
    """Create sample enrichment result."""
    return EnrichmentV1(
        enrichment_version="v1",
        metadata=EnrichmentMetadata(
            provider="openai",
            model="gpt-4-turbo",
            timestamp=datetime.utcnow(),
            cost_usd=0.015,
            tokens_used=500,
            enrichment_types=["summary"]
        ),
        summary={"short": "Test", "medium": "Test", "long": "Test"}
    )


class TestFilenameGeneration:
    """Test suite for filename generation."""
    
    def test_generate_output_filename_default(self):
        """Test default output filename generation."""
        filename = generate_output_filename("transcript.json")
        
        assert filename == "transcript-enriched.json"
    
    def test_generate_output_filename_custom_suffix(self):
        """Test output filename with custom suffix."""
        filename = generate_output_filename("transcript.json", suffix="-processed")
        
        assert filename == "transcript-processed.json"
    
    def test_generate_output_filename_no_extension(self):
        """Test output filename without extension."""
        filename = generate_output_filename("transcript")
        
        assert filename == "transcript-enriched"
    
    def test_generate_output_filename_multiple_dots(self):
        """Test output filename with multiple dots."""
        filename = generate_output_filename("my.transcript.json")
        
        assert filename == "my.transcript-enriched.json"


class TestPathResolution:
    """Test suite for output path resolution."""
    
    def test_resolve_output_path_explicit(self):
        """Test path resolution with explicit output path."""
        path = resolve_output_path(
            input_path="/input/transcript.json",
            output_path="/output/result.json"
        )
        
        assert path == Path("/output/result.json")
    
    def test_resolve_output_path_with_dir(self):
        """Test path resolution with output directory."""
        path = resolve_output_path(
            input_path="/input/transcript.json",
            output_dir="/output"
        )
        
        assert path == Path("/output/transcript-enriched.json")
    
    def test_resolve_output_path_default(self):
        """Test path resolution with default settings."""
        path = resolve_output_path(
            input_path="/input/transcript.json"
        )
        
        assert path == Path("/input/transcript-enriched.json")
    
    def test_resolve_output_path_priority(self):
        """Test that explicit path takes priority over directory."""
        path = resolve_output_path(
            input_path="/input/transcript.json",
            output_path="/explicit/result.json",
            output_dir="/dir"
        )
        
        # Explicit path should take priority
        assert path == Path("/explicit/result.json")


class TestPathValidation:
    """Test suite for output path validation."""
    
    def test_validate_output_path_valid(self, tmp_path):
        """Test validation of valid output path."""
        output_path = tmp_path / "output.json"
        
        # Should not raise
        validate_output_path(output_path, create_dirs=True)
    
    def test_validate_output_path_creates_dirs(self, tmp_path):
        """Test that validation creates parent directories."""
        output_path = tmp_path / "subdir" / "output.json"
        
        validate_output_path(output_path, create_dirs=True)
        
        # Parent directory should be created
        assert output_path.parent.exists()
    
    def test_validate_output_path_no_create_dirs(self, tmp_path):
        """Test validation without creating directories."""
        output_path = tmp_path / "nonexistent" / "output.json"
        
        with pytest.raises(OutputFileError) as exc_info:
            validate_output_path(output_path, create_dirs=False)
        
        assert "does not exist" in str(exc_info.value)
    
    @pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't work the same on Windows")
    def test_validate_output_path_not_writable(self, tmp_path):
        """Test validation of non-writable path."""
        output_path = tmp_path / "output.json"
        
        # Make parent directory read-only
        tmp_path.chmod(0o444)
        
        try:
            with pytest.raises(OutputFileError) as exc_info:
                validate_output_path(output_path, create_dirs=False)
            
            assert "not writable" in str(exc_info.value)
        finally:
            # Restore permissions
            tmp_path.chmod(0o755)


class TestOverwriteCheck:
    """Test suite for overwrite checking."""
    
    def test_check_overwrite_file_not_exists(self, tmp_path):
        """Test overwrite check when file doesn't exist."""
        output_path = tmp_path / "output.json"
        
        should_overwrite = check_overwrite(output_path, force=False)
        
        assert should_overwrite is True
    
    def test_check_overwrite_force(self, tmp_path):
        """Test overwrite check with force flag."""
        output_path = tmp_path / "output.json"
        output_path.touch()  # Create file
        
        should_overwrite = check_overwrite(output_path, force=True)
        
        assert should_overwrite is True
    
    @patch('builtins.input', return_value='y')
    def test_check_overwrite_user_confirms(self, mock_input, tmp_path):
        """Test overwrite check when user confirms."""
        output_path = tmp_path / "output.json"
        output_path.touch()  # Create file
        
        should_overwrite = check_overwrite(output_path, force=False)
        
        assert should_overwrite is True
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='n')
    def test_check_overwrite_user_declines(self, mock_input, tmp_path):
        """Test overwrite check when user declines."""
        output_path = tmp_path / "output.json"
        output_path.touch()  # Create file
        
        should_overwrite = check_overwrite(output_path, force=False)
        
        assert should_overwrite is False
        mock_input.assert_called_once()


class TestWriteEnrichmentResult:
    """Test suite for writing enrichment results."""
    
    def test_write_enrichment_result_success(self, tmp_path, sample_enrichment):
        """Test successful write of enrichment result."""
        output_path = tmp_path / "output.json"
        
        write_enrichment_result(sample_enrichment, output_path, force=True)
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["enrichment_version"] == "v1"
        assert data["metadata"]["provider"] == "openai"
    
    def test_write_enrichment_result_creates_dirs(self, tmp_path, sample_enrichment):
        """Test that write creates parent directories."""
        output_path = tmp_path / "subdir" / "output.json"
        
        write_enrichment_result(sample_enrichment, output_path, force=True)
        
        # Verify file was created
        assert output_path.exists()
    
    def test_write_enrichment_result_custom_indent(self, tmp_path, sample_enrichment):
        """Test write with custom indentation."""
        output_path = tmp_path / "output.json"
        
        write_enrichment_result(sample_enrichment, output_path, indent=4, force=True)
        
        # Verify indentation
        with open(output_path, 'r') as f:
            content = f.read()
        
        # 4-space indentation should be present
        assert "    " in content
    
    @patch('builtins.input', return_value='n')
    def test_write_enrichment_result_user_cancels(self, mock_input, tmp_path, sample_enrichment):
        """Test write when user cancels overwrite."""
        output_path = tmp_path / "output.json"
        output_path.touch()  # Create existing file
        
        with pytest.raises(OutputFileError) as exc_info:
            write_enrichment_result(sample_enrichment, output_path, force=False)
        
        assert "cancelled" in str(exc_info.value)


class TestWriteBatchResults:
    """Test suite for writing batch results."""
    
    def test_write_batch_results_success(self, tmp_path, sample_enrichment):
        """Test successful write of batch results."""
        output_dir = tmp_path / "output"
        
        results = {
            "/input/file1.json": sample_enrichment,
            "/input/file2.json": sample_enrichment
        }
        
        stats = write_batch_results(results, output_dir, force=True)
        
        # Verify statistics
        assert stats["success_count"] == 2
        assert stats["failed_count"] == 0
        assert len(stats["failed_files"]) == 0
        
        # Verify files were created
        assert (output_dir / "file1-enriched.json").exists()
        assert (output_dir / "file2-enriched.json").exists()
    
    def test_write_batch_results_with_failures(self, tmp_path, sample_enrichment):
        """Test batch write with some failures."""
        output_dir = tmp_path / "output"
        
        # Create invalid enrichment that will fail serialization
        invalid_enrichment = Mock()
        invalid_enrichment.model_dump.side_effect = Exception("Serialization error")
        
        results = {
            "/input/file1.json": sample_enrichment,
            "/input/file2.json": invalid_enrichment
        }
        
        stats = write_batch_results(results, output_dir, force=True)
        
        # Verify statistics
        assert stats["success_count"] == 1
        assert stats["failed_count"] == 1
        assert len(stats["failed_files"]) == 1
        assert stats["failed_files"][0]["input_path"] == "/input/file2.json"
    
    def test_write_batch_results_creates_output_dir(self, tmp_path, sample_enrichment):
        """Test that batch write creates output directory."""
        output_dir = tmp_path / "nonexistent" / "output"
        
        results = {
            "/input/file1.json": sample_enrichment
        }
        
        write_batch_results(results, output_dir, force=True)
        
        # Verify directory was created
        assert output_dir.exists()


class TestOutputManager:
    """Test suite for OutputManager."""
    
    def test_output_manager_initialization(self):
        """Test output manager initialization."""
        manager = OutputManager(force_overwrite=True)
        
        assert manager.force_overwrite is True
    
    def test_output_manager_prepare_output(self, tmp_path):
        """Test output path preparation."""
        manager = OutputManager()
        
        output_path = manager.prepare_output(
            input_path=str(tmp_path / "input.json"),
            output_dir=str(tmp_path / "output")
        )
        
        assert isinstance(output_path, Path)
        assert output_path.parent.exists()
    
    def test_output_manager_write_result(self, tmp_path, sample_enrichment):
        """Test writing result through manager."""
        manager = OutputManager(force_overwrite=True)
        output_path = tmp_path / "output.json"
        
        manager.write_result(sample_enrichment, output_path)
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["enrichment_version"] == "v1"
    
    def test_output_manager_workflow(self, tmp_path, sample_enrichment):
        """Test complete output manager workflow."""
        manager = OutputManager(force_overwrite=True)
        
        # Prepare output path
        output_path = manager.prepare_output(
            input_path=str(tmp_path / "input.json"),
            output_dir=str(tmp_path / "output")
        )
        
        # Write result
        manager.write_result(sample_enrichment, output_path)
        
        # Verify file exists
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["provider"] == "openai"


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_output_file_error_attributes(self):
        """Test OutputFileError attributes."""
        error = OutputFileError("Test error message")
        
        assert str(error) == "Test error message"
    
    @pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't work the same on Windows")
    def test_validate_output_path_permission_error(self, tmp_path):
        """Test validation with permission error."""
        output_path = tmp_path / "readonly" / "output.json"
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)
        
        try:
            with pytest.raises(OutputFileError) as exc_info:
                validate_output_path(output_path, create_dirs=False)
            
            assert "not writable" in str(exc_info.value)
        finally:
            # Restore permissions
            readonly_dir.chmod(0o755)
    
    @pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't work the same on Windows")
    def test_write_enrichment_result_write_error(self, tmp_path, sample_enrichment):
        """Test write with file write error."""
        output_path = tmp_path / "output.json"
        
        # Create file and make it read-only
        output_path.touch()
        output_path.chmod(0o444)
        
        try:
            with pytest.raises(OutputFileError) as exc_info:
                write_enrichment_result(sample_enrichment, output_path, force=True)
            
            assert "Failed to write" in str(exc_info.value)
        finally:
            # Restore permissions
            output_path.chmod(0o644)
