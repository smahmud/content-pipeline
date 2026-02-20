"""
Integration tests for enrich command separate output files feature (v0.8.6).

Tests the new default behavior where each enrichment type is saved to
a separate file, and the --combine flag for backward compatibility.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from cli.enrich import enrich
from pipeline.enrichment.schemas import (
    IndividualEnrichment,
    IndividualEnrichmentMetadata,
    extract_individual_enrichment,
    EnrichmentV1,
    EnrichmentMetadata,
    SummaryEnrichment,
    TagEnrichment,
)
from pipeline.llm.providers.base import LLMResponse
from tests.fixtures.mock_llm_responses import MOCK_SUMMARY_RESPONSE


class TestIndividualEnrichmentSchema:
    """Tests for IndividualEnrichment schema."""
    
    def test_individual_enrichment_creation(self):
        """Test creating an IndividualEnrichment instance."""
        metadata = IndividualEnrichmentMetadata(
            provider="cloud-openai",
            model="gpt-4-turbo",
            cost_usd=0.05,
            tokens_used=1000
        )
        
        individual = IndividualEnrichment(
            enrichment_type="summary",
            metadata=metadata,
            data={"short": "Brief", "medium": "Medium", "long": "Long"}
        )
        
        assert individual.enrichment_version == "v1"
        assert individual.enrichment_type == "summary"
        assert individual.metadata.provider == "cloud-openai"
        assert individual.data["short"] == "Brief"
    
    def test_individual_enrichment_serialization(self):
        """Test IndividualEnrichment serializes to JSON correctly."""
        metadata = IndividualEnrichmentMetadata(
            provider="cloud-openai",
            model="gpt-4-turbo",
            cost_usd=0.05,
            tokens_used=1000
        )
        
        individual = IndividualEnrichment(
            enrichment_type="tag",
            metadata=metadata,
            data={"categories": ["Tech"], "keywords": ["AI"], "entities": ["OpenAI"]}
        )
        
        json_data = individual.model_dump(mode='json')
        
        assert json_data["enrichment_version"] == "v1"
        assert json_data["enrichment_type"] == "tag"
        assert json_data["metadata"]["provider"] == "cloud-openai"
        assert json_data["data"]["categories"] == ["Tech"]


class TestExtractIndividualEnrichment:
    """Tests for extract_individual_enrichment function."""
    
    @pytest.fixture
    def full_enrichment(self):
        """Create a full EnrichmentV1 with all types."""
        from datetime import datetime
        
        return EnrichmentV1(
            metadata=EnrichmentMetadata(
                provider="cloud-openai",
                model="gpt-4-turbo",
                timestamp=datetime.utcnow(),
                cost_usd=0.10,
                tokens_used=2000,
                enrichment_types=["summary", "tag"]
            ),
            summary=SummaryEnrichment(
                short="Short summary",
                medium="Medium summary",
                long="Long summary"
            ),
            tags=TagEnrichment(
                categories=["Technology"],
                keywords=["AI", "ML"],
                entities=["OpenAI"]
            )
        )
    
    def test_extract_summary(self, full_enrichment):
        """Test extracting summary from EnrichmentV1."""
        individual = extract_individual_enrichment(full_enrichment, "summary")
        
        assert individual.enrichment_type == "summary"
        assert individual.data["short"] == "Short summary"
        assert individual.metadata.provider == "cloud-openai"
    
    def test_extract_tag(self, full_enrichment):
        """Test extracting tags from EnrichmentV1."""
        individual = extract_individual_enrichment(full_enrichment, "tag")
        
        assert individual.enrichment_type == "tag"
        assert individual.data["categories"] == ["Technology"]
        assert individual.data["keywords"] == ["AI", "ML"]
    
    def test_extract_missing_type_raises(self, full_enrichment):
        """Test extracting missing type raises ValueError."""
        with pytest.raises(ValueError, match="not present"):
            extract_individual_enrichment(full_enrichment, "chapter")
    
    def test_extract_unknown_type_raises(self, full_enrichment):
        """Test extracting unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown enrichment type"):
            extract_individual_enrichment(full_enrichment, "unknown")


class TestSeparateOutputMode:
    """Tests for separate output file mode."""
    
    @pytest.fixture
    def sample_transcript(self, tmp_path):
        """Create a sample transcript file."""
        transcript = {
            "metadata": {"language": "en", "duration": 300.0},
            "transcript": [{"text": "This is test content for enrichment."}]
        }
        transcript_path = tmp_path / "test_transcript.json"
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f)
        return transcript_path
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = Mock()
        
        # Configure capabilities
        mock_provider.get_capabilities.return_value = {
            "provider": "openai",
            "models": ["gpt-4-turbo"],
            "max_tokens": 128000,
            "default_model": "gpt-4-turbo"
        }
        
        # Configure validation
        mock_provider.validate_requirements.return_value = True
        
        # Configure cost estimation (returns simple float)
        mock_provider.estimate_cost.return_value = 0.01
        
        # Configure context window for chunking
        mock_provider.get_context_window.return_value = 128000
        
        # Configure generate method
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(MOCK_SUMMARY_RESPONSE),
            model_used="gpt-4-turbo",
            tokens_used=500,
            cost_usd=0.015
        )
        
        return mock_provider
    
    @patch('cli.enrich.LLMProviderFactory')
    def test_separate_mode_creates_type_specific_files(
        self, mock_factory_class, sample_transcript, tmp_path, mock_llm_provider
    ):
        """Test that single type creates type-specific output file.
        
        Note: Single type uses combined mode internally but with type-specific filename.
        The output is still EnrichmentV1 format, not IndividualEnrichment.
        """
        mock_factory = Mock()
        mock_factory.create_provider.return_value = mock_llm_provider
        mock_factory_class.return_value = mock_factory
        
        runner = CliRunner()
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output-dir', str(tmp_path),
            '--provider', 'openai',
            '--summarize'
        ])
        
        assert result.exit_code == 0
        
        # Check type-specific file was created
        summary_file = tmp_path / "test_transcript-summary.json"
        assert summary_file.exists()
        
        # Verify content structure (EnrichmentV1 format for single type)
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        assert data["enrichment_version"] == "v1"
        assert "summary" in data
        assert "metadata" in data
    
    @patch('cli.enrich.LLMProviderFactory')
    def test_combine_flag_creates_single_file(
        self, mock_factory_class, sample_transcript, tmp_path, mock_llm_provider
    ):
        """Test that --combine flag creates single combined file."""
        mock_factory = Mock()
        mock_factory.create_provider.return_value = mock_llm_provider
        mock_factory_class.return_value = mock_factory
        
        runner = CliRunner()
        output_path = tmp_path / "combined.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize',
            '--combine'
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify combined structure (EnrichmentV1)
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["enrichment_version"] == "v1"
        assert "summary" in data
        assert "metadata" in data
    
    @patch('cli.enrich.LLMProviderFactory')
    def test_output_without_combine_multiple_types_fails(
        self, mock_factory_class, sample_transcript, tmp_path, mock_llm_provider
    ):
        """Test that --output without --combine fails for multiple types."""
        mock_factory = Mock()
        mock_factory.create_provider.return_value = mock_llm_provider
        mock_factory_class.return_value = mock_factory
        
        runner = CliRunner()
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize',
            '--tag'
        ])
        
        assert result.exit_code == 1
        assert "cannot be used with multiple enrichment types" in result.output
    
    @patch('cli.enrich.LLMProviderFactory')
    def test_output_dir_creates_directory(
        self, mock_factory_class, sample_transcript, tmp_path, mock_llm_provider
    ):
        """Test that --output-dir creates directory if it doesn't exist."""
        mock_factory = Mock()
        mock_factory.create_provider.return_value = mock_llm_provider
        mock_factory_class.return_value = mock_factory
        
        runner = CliRunner()
        output_dir = tmp_path / "new_dir" / "nested"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output-dir', str(output_dir),
            '--provider', 'openai',
            '--summarize'
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "test_transcript-summary.json").exists()


class TestOutputFileNaming:
    """Tests for output file naming conventions."""
    
    def test_summary_suffix(self):
        """Test summary type creates -summary.json suffix."""
        from cli.enrich import _generate_output_path_for_type
        
        result = _generate_output_path_for_type("transcript.json", "summary")
        assert result.endswith("-summary.json")
    
    def test_tag_suffix(self):
        """Test tag type creates -tags.json suffix (pluralized)."""
        from cli.enrich import _generate_output_path_for_type
        
        result = _generate_output_path_for_type("transcript.json", "tag")
        assert result.endswith("-tags.json")
    
    def test_chapter_suffix(self):
        """Test chapter type creates -chapters.json suffix."""
        from cli.enrich import _generate_output_path_for_type
        
        result = _generate_output_path_for_type("transcript.json", "chapter")
        assert result.endswith("-chapters.json")
    
    def test_highlight_suffix(self):
        """Test highlight type creates -highlights.json suffix."""
        from cli.enrich import _generate_output_path_for_type
        
        result = _generate_output_path_for_type("transcript.json", "highlight")
        assert result.endswith("-highlights.json")
    
    def test_output_dir_respected(self):
        """Test output_dir is used in path generation."""
        from cli.enrich import _generate_output_path_for_type
        
        result = _generate_output_path_for_type(
            "data/transcript.json",
            "summary",
            output_dir="output"
        )
        # Use Path for cross-platform comparison
        assert Path(result).name == "transcript-summary.json"
        assert "output" in result
