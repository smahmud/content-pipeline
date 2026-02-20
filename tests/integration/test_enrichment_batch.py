"""
Integration tests for batch enrichment processing

Tests batch processing of multiple transcript files with progress tracking,
cost estimation, error handling, and summary reporting.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.enrichment.batch import BatchProcessor, BatchReport, BatchResult
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest
from pipeline.llm.providers.base import LLMResponse
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata
from pipeline.enrichment.errors import CostLimitExceededError
from tests.fixtures.mock_llm_responses import (
    MOCK_SUMMARY_RESPONSE,
    MOCK_TAG_RESPONSE
)
from datetime import datetime


@pytest.fixture
def sample_transcripts(tmp_path):
    """Create multiple sample transcript files for batch testing."""
    transcripts = []
    
    for i in range(3):
        transcript = {
            "transcript_version": "v1",
            "text": f"This is test transcript {i+1}. It contains sample content for testing batch processing.",
            "metadata": {
                "language": "en",
                "duration": 60.0 + (i * 30),
                "source": f"test_audio_{i+1}.mp3"
            }
        }
        
        transcript_path = tmp_path / f"transcript_{i+1}.json"
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f)
        
        transcripts.append(transcript_path)
    
    return transcripts


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for batch testing."""
    orchestrator = Mock(spec=EnrichmentOrchestrator)
    
    # Configure provider factory
    orchestrator.provider_factory = Mock()
    provider = Mock()
    provider.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-4-turbo"],
        "max_tokens": 128000,
        "default_model": "gpt-4-turbo"
    }
    provider.estimate_cost.return_value = 0.05
    orchestrator.provider_factory.create_provider.return_value = provider
    
    # Configure enrich method to return successful results
    def mock_enrich(request):
        # Check cost limit
        if request.max_cost and request.max_cost < 0.05:
            raise CostLimitExceededError(
                f"Estimated cost $0.05 exceeds limit ${request.max_cost}"
            )
        
        return EnrichmentV1(
            enrichment_version="v1",
            metadata=EnrichmentMetadata(
                provider="openai",
                model="gpt-4-turbo",
                timestamp=datetime.utcnow(),
                cost_usd=0.05,
                tokens_used=1000,
                enrichment_types=request.enrichment_types
            ),
            summary=MOCK_SUMMARY_RESPONSE if "summary" in request.enrichment_types else None,
            tags=MOCK_TAG_RESPONSE if "tag" in request.enrichment_types else None
        )
    
    orchestrator.enrich.side_effect = mock_enrich
    
    return orchestrator


class TestBatchProcessing:
    """Integration tests for batch enrichment processing."""
    
    def test_basic_batch_processing(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test basic batch processing of multiple files."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
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
        
        # Verify all files were processed
        assert len(report.results) == 3
        for result in report.results:
            assert result.success
            assert result.cost > 0
            assert result.tokens > 0
    
    def test_batch_with_output_directory(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing with custom output directory."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Create output directory
        output_dir = tmp_path / "enriched_output"
        output_dir.mkdir()
        
        # Process batch
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai",
            output_dir=str(output_dir)
        )
        
        # Verify all output files were created in output directory
        assert report.successful == 3
        
        for i in range(3):
            output_file = output_dir / f"transcript_{i+1}-enriched.json"
            assert output_file.exists()
            
            # Verify file content
            with open(output_file, 'r') as f:
                enrichment = json.load(f)
            assert enrichment['enrichment_version'] == 'v1'
    
    def test_batch_cost_limit_enforcement(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch cost limit prevents processing."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch with very low cost limit
        pattern = str(tmp_path / "transcript_*.json")
        
        with pytest.raises(CostLimitExceededError) as exc_info:
            processor.process_batch(
                pattern=pattern,
                enrichment_types=["summary"],
                provider="openai",
                max_cost=0.01  # Very low limit
            )
        
        assert "exceeds limit" in str(exc_info.value)
    
    def test_batch_individual_file_failure(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing continues after individual file failure."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Configure orchestrator to fail on second file
        call_count = [0]
        
        def mock_enrich_with_failure(request):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated enrichment failure")
            
            return EnrichmentV1(
                enrichment_version="v1",
                metadata=EnrichmentMetadata(
                    provider="openai",
                    model="gpt-4-turbo",
                    timestamp=datetime.utcnow(),
                    cost_usd=0.05,
                    tokens_used=1000,
                    enrichment_types=request.enrichment_types
                ),
                summary=MOCK_SUMMARY_RESPONSE
            )
        
        mock_orchestrator.enrich.side_effect = mock_enrich_with_failure
        
        # Process batch
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Verify report shows partial success
        assert report.total_files == 3
        assert report.successful == 2
        assert report.failed == 1
        
        # Verify failed result has error message
        failed_results = [r for r in report.results if not r.success]
        assert len(failed_results) == 1
        assert failed_results[0].error is not None
    
    def test_batch_with_caching(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing with caching enabled."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch with caching
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai",
            use_cache=True
        )
        
        # Verify all files were processed
        assert report.successful == 3
        
        # Verify enrich was called with use_cache=True
        for call in mock_orchestrator.enrich.call_args_list:
            request = call[0][0]
            assert request.use_cache is True
    
    def test_batch_without_caching(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing with caching disabled."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch without caching
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai",
            use_cache=False
        )
        
        # Verify all files were processed
        assert report.successful == 3
        
        # Verify enrich was called with use_cache=False
        for call in mock_orchestrator.enrich.call_args_list:
            request = call[0][0]
            assert request.use_cache is False
    
    def test_batch_multiple_enrichment_types(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing with multiple enrichment types."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch with multiple types
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary", "tag"],
            provider="openai"
        )
        
        # Verify all files were processed
        assert report.successful == 3
        
        # Verify enrich was called with both types
        for call in mock_orchestrator.enrich.call_args_list:
            request = call[0][0]
            assert "summary" in request.enrichment_types
            assert "tag" in request.enrichment_types
    
    def test_batch_no_matching_files(self, tmp_path, mock_orchestrator):
        """Test batch processing with no matching files."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Try to process non-existent files
        pattern = str(tmp_path / "nonexistent_*.json")
        
        with pytest.raises(ValueError) as exc_info:
            processor.process_batch(
                pattern=pattern,
                enrichment_types=["summary"],
                provider="openai"
            )
        
        assert "No files found" in str(exc_info.value)
    
    def test_batch_report_formatting(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch report formatting."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Format report
        formatted = processor.format_report(report)
        
        # Verify formatted output
        assert "Batch Processing Report" in formatted
        assert f"Total files: {report.total_files}" in formatted
        assert f"Successful: {report.successful}" in formatted
        assert f"Failed: {report.failed}" in formatted
        assert f"Total cost: ${report.total_cost:.4f}" in formatted
    
    def test_batch_with_specific_model(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing with specific model selection."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch with specific model
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        # Verify all files were processed
        assert report.successful == 3
        
        # Verify enrich was called with specific model
        for call in mock_orchestrator.enrich.call_args_list:
            request = call[0][0]
            assert request.model == "gpt-3.5-turbo"
    
    def test_batch_timing_metrics(self, sample_transcripts, mock_orchestrator, tmp_path):
        """Test batch processing tracks timing metrics."""
        processor = BatchProcessor(mock_orchestrator)
        
        # Process batch
        pattern = str(tmp_path / "transcript_*.json")
        report = processor.process_batch(
            pattern=pattern,
            enrichment_types=["summary"],
            provider="openai"
        )
        
        # Verify timing metrics (mocks may return instantly, so >= 0 is valid)
        assert report.total_duration >= 0
        
        for result in report.results:
            assert result.duration >= 0
