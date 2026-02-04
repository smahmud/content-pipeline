"""
Integration tests for enrichment CLI workflow

Tests end-to-end CLI workflow from command invocation to output file creation.
Uses mocked LLM responses to avoid actual API calls while testing the complete
pipeline including file I/O, caching, error handling, and result formatting.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from cli.enrich import enrich
from pipeline.llm.providers.base import LLMResponse
from tests.fixtures.mock_llm_responses import (
    MOCK_SUMMARY_RESPONSE,
    MOCK_TAG_RESPONSE,
    MOCK_CHAPTERS_RESPONSE,
    MOCK_HIGHLIGHTS_RESPONSE
)


@pytest.fixture
def sample_transcript(tmp_path):
    """Create a sample transcript file for testing."""
    transcript = {
        "transcript_version": "v1",
        "text": "This is a test transcript about machine learning. "
                "It discusses neural networks and deep learning concepts. "
                "The content covers supervised and unsupervised learning approaches.",
        "metadata": {
            "language": "en",
            "duration": 180.0,
            "source": "test_audio.mp3"
        }
    }
    
    transcript_path = tmp_path / "test_transcript.json"
    with open(transcript_path, 'w') as f:
        json.dump(transcript, f)
    
    return transcript_path


@pytest.fixture
def mock_llm_agent():
    """Create a mock LLM agent that returns realistic responses."""
    agent = Mock()
    
    # Configure capabilities
    agent.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-4-turbo"],
        "max_tokens": 128000
    }
    
    # Configure validation
    agent.validate_requirements.return_value = True
    
    # Configure generate method to return different responses based on call count
    responses = [
        LLMResponse(
            content=json.dumps(MOCK_SUMMARY_RESPONSE),
            model_used="gpt-4-turbo",
            tokens_used=500,
            cost_usd=0.015
        ),
        LLMResponse(
            content=json.dumps(MOCK_TAG_RESPONSE),
            model_used="gpt-4-turbo",
            tokens_used=300,
            cost_usd=0.009
        ),
        LLMResponse(
            content=json.dumps(MOCK_CHAPTERS_RESPONSE),
            model_used="gpt-4-turbo",
            tokens_used=600,
            cost_usd=0.018
        ),
        LLMResponse(
            content=json.dumps(MOCK_HIGHLIGHTS_RESPONSE),
            model_used="gpt-4-turbo",
            tokens_used=400,
            cost_usd=0.012
        )
    ]
    
    agent.generate.side_effect = responses
    
    return agent


class TestEnrichmentCLIWorkflow:
    """Integration tests for complete CLI workflow."""
    
    @patch('cli.enrich.AgentFactory')
    def test_basic_enrichment_workflow(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test basic enrichment workflow from CLI to output file."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command
        runner = CliRunner()
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        assert "Enrichment Completed Successfully" in result.output
        
        # Verify output file was created
        assert output_path.exists()
        
        # Verify output file content
        with open(output_path, 'r') as f:
            enrichment = json.load(f)
        
        assert enrichment['enrichment_version'] == 'v1'
        assert enrichment['metadata']['provider'] == 'openai'
        assert enrichment['metadata']['model'] == 'gpt-4-turbo'
        assert enrichment['summary'] is not None
        assert 'short' in enrichment['summary']
        assert 'medium' in enrichment['summary']
        assert 'long' in enrichment['summary']
    
    @patch('cli.enrich.AgentFactory')
    def test_all_enrichment_types(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test enrichment with all types enabled."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with --all flag
        runner = CliRunner()
        output_path = tmp_path / "output_all.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--all'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify output file content
        with open(output_path, 'r') as f:
            enrichment = json.load(f)
        
        # Verify all enrichment types are present
        assert enrichment['summary'] is not None
        assert enrichment['tags'] is not None
        assert enrichment['chapters'] is not None
        assert enrichment['highlights'] is not None
        
        # Verify metadata includes all types
        assert set(enrichment['metadata']['enrichment_types']) == {
            'summary', 'tag', 'chapter', 'highlight'
        }
    
    @patch('cli.enrich.AgentFactory')
    def test_auto_output_path_generation(self, mock_factory_class, sample_transcript, mock_llm_agent):
        """Test automatic output path generation when not specified."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command without --output
        runner = CliRunner()
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--provider', 'openai',
            '--summarize'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify output file was created with auto-generated name
        expected_output = sample_transcript.parent / "test_transcript-enriched.json"
        assert expected_output.exists()
        
        # Cleanup
        expected_output.unlink()
    
    @patch('cli.enrich.AgentFactory')
    def test_dry_run_mode(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test dry-run mode displays cost estimate without execution."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with --dry-run
        runner = CliRunner()
        output_path = tmp_path / "output_dryrun.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize',
            '--dry-run'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify dry-run output
        assert "DRY RUN - Cost Estimate" in result.output
        assert "Estimated cost:" in result.output
        assert "No API calls were made" in result.output
        
        # Verify no output file was created
        assert not output_path.exists()
        
        # Verify agent generate was not called
        mock_llm_agent.generate.assert_not_called()
    
    @patch('cli.enrich.AgentFactory')
    def test_cost_limit_enforcement(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test cost limit enforcement prevents execution."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with very low cost limit
        runner = CliRunner()
        output_path = tmp_path / "output_costlimit.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize',
            '--max-cost', '0.001'
        ])
        
        # Verify command failed with cost limit error
        assert result.exit_code == 1
        assert "Cost Limit Exceeded" in result.output
        
        # Verify no output file was created
        assert not output_path.exists()
    
    @patch('cli.enrich.AgentFactory')
    def test_no_enrichment_types_error(self, mock_factory_class, sample_transcript, tmp_path):
        """Test error when no enrichment types are specified."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command without any enrichment type flags
        runner = CliRunner()
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--provider', 'openai'
        ])
        
        # Verify command failed
        assert result.exit_code == 1
        assert "No enrichment types specified" in result.output
    
    @patch('cli.enrich.AgentFactory')
    def test_invalid_transcript_file(self, mock_factory_class, tmp_path):
        """Test error handling for invalid transcript file."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Create invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("not valid json {")
        
        # Run CLI command
        runner = CliRunner()
        
        result = runner.invoke(enrich, [
            '--input', str(invalid_file),
            '--provider', 'openai',
            '--summarize'
        ])
        
        # Verify command failed
        assert result.exit_code == 1
    
    @patch('cli.enrich.AgentFactory')
    def test_provider_auto_selection(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test auto provider selection."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with auto provider
        runner = CliRunner()
        output_path = tmp_path / "output_auto.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'auto',
            '--summarize'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify factory was called with "auto"
        mock_factory.create_agent.assert_called_with("auto")
    
    @patch('cli.enrich.AgentFactory')
    def test_specific_model_selection(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test specific model selection overrides default."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with specific model
        runner = CliRunner()
        output_path = tmp_path / "output_model.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--model', 'gpt-3.5-turbo',
            '--summarize'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify output file content
        with open(output_path, 'r') as f:
            enrichment = json.load(f)
        
        # Note: Model in output will be what the mock returns, but request should include it
        assert enrichment['enrichment_version'] == 'v1'
    
    @patch('cli.enrich.AgentFactory')
    def test_cache_bypass(self, mock_factory_class, sample_transcript, tmp_path, mock_llm_agent):
        """Test --no-cache flag bypasses cache."""
        # Setup mock factory
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_llm_agent
        mock_factory_class.return_value = mock_factory
        
        # Run CLI command with --no-cache
        runner = CliRunner()
        output_path = tmp_path / "output_nocache.json"
        
        result = runner.invoke(enrich, [
            '--input', str(sample_transcript),
            '--output', str(output_path),
            '--provider', 'openai',
            '--summarize',
            '--no-cache'
        ])
        
        # Verify command succeeded
        assert result.exit_code == 0
        
        # Verify output file was created
        assert output_path.exists()
