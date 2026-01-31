"""
Unit tests for BatchProcessor

Tests batch processing, progress tracking, cost estimation, and error handling.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline.enrichment.batch import (
    BatchProcessor,
    BatchResult,
    BatchReport
)
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata
from pipeline.enrichment.errors import CostLimitExceededError
from datetime import datetime


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    orchestrator = Mock(spec=EnrichmentOrchestrator)
    
    # Configure agent factory
    orchestrator.agent_factory = Mock()
    agent = Mock()
    agent.get_context_window.return_value = 8000
    agent.estimate_cost.return_value = 0.050  # High cost to trigger limit
    orchestrator.agent_factory.create_agent.return_value = agent
    
    # Configure enrich method
    orchestrator.enrich.return_value = EnrichmentV1(
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
    
    return orchestrator


@pytest.fixture
def batch_processor(mock_orchestrator):
    """Create batch processor with mock orchestrator."""
    return BatchProcessor(orchestrator=mock_orchestrator)


@pytest.fixture
def temp_transcript_files(tmp_path):
    """Create temporary transcript files for testing."""
    files = []
    
    for i in range(3):
        file_path = tmp_path / f"transcript_{i}.json"
        data = {
            "text": f"This is transcript {i}.",
            "metadata": {
                "language": "en",
                "duration": 60.0 + i * 10
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        files.append(str(file_path))
    
    return files


class TestBatchProcessor:
    """Test suite for BatchProcessor."""
    
    def test_initialization(self, mock_orchestrator):
        """Test batch processor initialization."""
        processor = BatchProcessor(orchestrator=mock_orchestrator)
        
        assert processor.orchestrator == mock_orchestrator
    
    def test_find_files(self, batch_processor, temp_transcript_files, tmp_path):
        """Test file discovery with glob pattern."""
        pattern = str(tmp_path / "transcript_*.json")
        
        files = batch_processor._find_files(pattern)
        
        assert len(files) == 3
        assert all(Path(f).exists() for f in files)
    
    def test_find_files_no_matches(self, batch_processor, tmp_path):
        """Test file discovery with no matches."""
        pattern = str(tmp_path / "nonexistent_*.json")
        
        files = batch_processor._find_files(pattern)
        
        assert len(files) == 0
    
    def test_process_batch_basic(self, batch_processor, temp_transcript_files, tmp_path):
        """Test basic batch processing."""
        pattern = str(tmp_path / "transcript_*.json")
        
        report = batch_processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Verify report
        assert isinstance(report, BatchReport)
        assert report.total_files == 3
        assert report.successful == 3
        assert report.failed == 0
        assert report.total_cost > 0
        assert report.total_tokens > 0
        assert report.total_duration > 0
    
    def test_process_batch_no_files(self, batch_processor, tmp_path):
        """Test batch processing with no matching files."""
        pattern = str(tmp_path / "nonexistent_*.json")
        
        with pytest.raises(ValueError) as exc_info:
            batch_processor.process_batch(
                pattern=pattern,
                enrichment_types=["summary"],
                provider="openai"
            )
        
        assert "No files found" in str(exc_info.value)
    
    def test_process_batch_with_cost_limit(self, batch_processor, temp_transcript_files, tmp_path):
        """Test batch processing with cost limit."""
        pattern = str(tmp_path / "transcript_*.json")
        
        # Set very low cost limit
        with pytest.raises(CostLimitExceededError) as exc_info:
            batch_processor.process_batch(
                pattern=pattern,
                enrichment_types=["summary"],
                provider="openai",
                max_cost=0.001
            )
        
        assert "exceeds limit" in str(exc_info.value)
    
    def test_process_batch_with_failures(self, batch_processor, mock_orchestrator, temp_transcript_files, tmp_path):
        """Test batch processing with some failures."""
        # Configure orchestrator to fail on second file
        mock_orchestrator.enrich.side_effect = [
            EnrichmentV1(
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
            ),
            Exception("Processing error"),
            EnrichmentV1(
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
        ]
        
        pattern = str(tmp_path / "transcript_*.json")
        
        report = batch_processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Verify report shows failures
        assert report.total_files == 3
        assert report.successful == 2
        assert report.failed == 1
    
    def test_generate_output_path_default(self, batch_processor):
        """Test output path generation with default settings."""
        input_path = "/path/to/transcript.json"
        
        output_path = batch_processor._generate_output_path(
            input_path=input_path,
            output_dir=None
        )
        
        # Use Path to normalize separators for cross-platform compatibility
        expected = str(Path("/path/to/transcript-enriched.json"))
        assert output_path == expected
    
    def test_generate_output_path_with_output_dir(self, batch_processor, tmp_path):
        """Test output path generation with output directory."""
        input_path = "/path/to/transcript.json"
        output_dir = str(tmp_path / "output")
        
        output_path = batch_processor._generate_output_path(
            input_path=input_path,
            output_dir=output_dir
        )
        
        assert output_path == str(tmp_path / "output" / "transcript-enriched.json")
    
    def test_save_result(self, batch_processor, tmp_path):
        """Test saving enrichment result."""
        result = EnrichmentV1(
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
        
        output_path = str(tmp_path / "output.json")
        
        batch_processor._save_result(result, output_path)
        
        # Verify file was created
        assert Path(output_path).exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["enrichment_version"] == "v1"
        assert data["metadata"]["provider"] == "openai"
    
    def test_format_report(self, batch_processor):
        """Test report formatting."""
        report = BatchReport(
            total_files=5,
            successful=4,
            failed=1,
            total_cost=0.060,
            total_tokens=2000,
            total_duration=45.5,
            results=[
                BatchResult(
                    input_path="/path/to/file1.json",
                    output_path="/path/to/file1-enriched.json",
                    success=True,
                    cost=0.015,
                    tokens=500,
                    duration=10.0
                ),
                BatchResult(
                    input_path="/path/to/file2.json",
                    output_path="",
                    success=False,
                    error="Processing error"
                )
            ]
        )
        
        formatted = batch_processor.format_report(report)
        
        # Verify formatted output
        assert "Total files: 5" in formatted
        assert "Successful: 4" in formatted
        assert "Failed: 1" in formatted
        assert "$0.0600" in formatted
        assert "2,000" in formatted
        assert "45.5s" in formatted
        assert "Failed files:" in formatted
        assert "/path/to/file2.json" in formatted
    
    def test_estimate_batch_cost(self, batch_processor, temp_transcript_files, tmp_path):
        """Test batch cost estimation."""
        pattern = str(tmp_path / "transcript_*.json")
        files = batch_processor._find_files(pattern)
        
        total_cost = batch_processor._estimate_batch_cost(
            input_files=files,
            enrichment_types=["summary"],
            provider="openai",
            model="gpt-4-turbo"
        )
        
        # Verify cost is estimated
        assert total_cost >= 0
    
    def test_process_single_file_success(self, batch_processor, temp_transcript_files, tmp_path):
        """Test processing single file successfully."""
        result = batch_processor._process_single_file(
            input_path=temp_transcript_files[0],
            enrichment_types=["summary"],
            provider="openai",
            output_dir=str(tmp_path / "output"),
            model="gpt-4-turbo",
            use_cache=True,
            custom_prompts_dir=None
        )
        
        # Verify result
        assert isinstance(result, BatchResult)
        assert result.success is True
        assert result.cost > 0
        assert result.tokens > 0
        assert result.duration > 0
        assert result.error is None
    
    def test_process_single_file_failure(self, batch_processor, mock_orchestrator, temp_transcript_files, tmp_path):
        """Test processing single file with failure."""
        # Configure orchestrator to fail
        mock_orchestrator.enrich.side_effect = Exception("Processing error")
        
        result = batch_processor._process_single_file(
            input_path=temp_transcript_files[0],
            enrichment_types=["summary"],
            provider="openai",
            output_dir=str(tmp_path / "output"),
            model="gpt-4-turbo",
            use_cache=True,
            custom_prompts_dir=None
        )
        
        # Verify result shows failure
        assert isinstance(result, BatchResult)
        assert result.success is False
        assert result.cost == 0.0
        assert result.tokens == 0
        assert result.error is not None
        assert "Processing error" in result.error
    
    @patch('pipeline.enrichment.batch.TQDM_AVAILABLE', True)
    @patch('pipeline.enrichment.batch.tqdm')
    def test_process_batch_with_tqdm(self, mock_tqdm, batch_processor, temp_transcript_files, tmp_path):
        """Test batch processing with tqdm progress bar."""
        # Configure tqdm mock
        mock_tqdm.return_value = temp_transcript_files
        
        pattern = str(tmp_path / "transcript_*.json")
        
        batch_processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Verify tqdm was called
        mock_tqdm.assert_called_once()
    
    def test_batch_result_dataclass(self):
        """Test BatchResult dataclass."""
        result = BatchResult(
            input_path="/path/to/input.json",
            output_path="/path/to/output.json",
            success=True,
            cost=0.015,
            tokens=500,
            duration=10.5
        )
        
        assert result.input_path == "/path/to/input.json"
        assert result.output_path == "/path/to/output.json"
        assert result.success is True
        assert result.cost == 0.015
        assert result.tokens == 500
        assert result.duration == 10.5
        assert result.error is None
    
    def test_batch_report_dataclass(self):
        """Test BatchReport dataclass."""
        report = BatchReport(
            total_files=10,
            successful=8,
            failed=2,
            total_cost=0.120,
            total_tokens=4000,
            total_duration=120.5,
            results=[]
        )
        
        assert report.total_files == 10
        assert report.successful == 8
        assert report.failed == 2
        assert report.total_cost == 0.120
        assert report.total_tokens == 4000
        assert report.total_duration == 120.5
        assert report.results == []
