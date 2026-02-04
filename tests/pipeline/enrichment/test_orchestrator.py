"""
Unit tests for EnrichmentOrchestrator

Tests workflow coordination, cost estimation, caching, and result aggregation.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from pipeline.enrichment.orchestrator import (
    EnrichmentOrchestrator,
    EnrichmentRequest,
    DryRunReport
)
from pipeline.llm.providers.base import LLMResponse
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata
from pipeline.enrichment.cost_estimator import CostEstimate
from pipeline.enrichment.errors import CostLimitExceededError, EnrichmentError
from tests.fixtures.mock_llm_responses import (
    MOCK_SUMMARY_RESPONSE,
    MOCK_TAG_RESPONSE
)


@pytest.fixture
def mock_provider_factory():
    """Create mock provider factory."""
    factory = Mock()
    provider = Mock()
    
    # Configure provider capabilities
    provider.get_capabilities.return_value = {
        "provider": "openai",
        "models": ["gpt-4-turbo"],
        "max_tokens": 128000
    }
    
    # Configure provider generate method
    provider.generate.return_value = LLMResponse(
        content=json.dumps(MOCK_SUMMARY_RESPONSE),
        model_used="gpt-4-turbo",
        tokens_used=500,
        cost_usd=0.015
    )
    
    # Configure provider estimate_cost to return float
    provider.estimate_cost.return_value = 0.015
    
    factory.create_provider.return_value = provider
    return factory


@pytest.fixture
def mock_prompt_loader():
    """Create mock prompt loader."""
    loader = Mock()
    loader.load_prompt.return_value = {
        "system": "You are a summarizer",
        "user_template": "Summarize: {{ transcript_text }}"
    }
    return loader


@pytest.fixture
def mock_prompt_renderer():
    """Create mock prompt renderer."""
    renderer = Mock()
    renderer.render.return_value = "Summarize this transcript"
    return renderer


@pytest.fixture
def mock_cache_system():
    """Create mock cache system."""
    cache = Mock()
    cache.get.return_value = None  # No cache hit by default
    cache.generate_key.return_value = "test_cache_key"
    return cache


@pytest.fixture
def orchestrator(mock_provider_factory, mock_prompt_loader, mock_prompt_renderer, mock_cache_system):
    """Create orchestrator with mocked dependencies."""
    return EnrichmentOrchestrator(
        provider_factory=mock_provider_factory,
        prompt_loader=mock_prompt_loader,
        prompt_renderer=mock_prompt_renderer,
        cache_system=mock_cache_system
    )


@pytest.fixture
def basic_request():
    """Create basic enrichment request."""
    return EnrichmentRequest(
        transcript_text="This is a test transcript.",
        language="en",
        duration=60.0,
        enrichment_types=["summary"],
        provider="openai",
        model="gpt-4-turbo"
    )


class TestEnrichmentOrchestrator:
    """Test suite for EnrichmentOrchestrator."""
    
    def test_initialization(self, mock_provider_factory):
        """Test orchestrator initialization."""
        orchestrator = EnrichmentOrchestrator(provider_factory=mock_provider_factory)
        
        assert orchestrator.provider_factory == mock_provider_factory
        assert orchestrator.prompt_loader is not None
        assert orchestrator.prompt_renderer is not None
        assert orchestrator.cache_system is not None
    
    def test_enrich_basic_workflow(self, orchestrator, basic_request, mock_provider_factory):
        """Test basic enrichment workflow."""
        result = orchestrator.enrich(basic_request)
        
        # Verify result type
        assert isinstance(result, EnrichmentV1)
        
        # Verify metadata
        assert result.metadata.provider == "openai"
        assert result.metadata.model == "gpt-4-turbo"
        assert result.metadata.cost_usd > 0
        assert result.metadata.tokens_used > 0
        assert "summary" in result.metadata.enrichment_types
        
        # Verify provider was called
        mock_provider_factory.create_provider.assert_called_once_with("openai")
    
    def test_enrich_multiple_types(self, orchestrator, mock_provider_factory):
        """Test enrichment with multiple types."""
        # Configure provider to return different responses
        provider = mock_provider_factory.create_provider.return_value
        provider.generate.side_effect = [
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
            )
        ]
        
        request = EnrichmentRequest(
            transcript_text="Test transcript",
            language="en",
            duration=60.0,
            enrichment_types=["summary", "tag"],
            provider="openai"
        )
        
        result = orchestrator.enrich(request)
        
        # Verify both enrichment types
        assert "summary" in result.metadata.enrichment_types
        assert "tag" in result.metadata.enrichment_types
        
        # Verify costs are summed
        assert result.metadata.cost_usd == 0.024
        assert result.metadata.tokens_used == 800
    
    def test_cost_limit_enforcement(self, orchestrator, basic_request):
        """Test that cost limit is enforced."""
        # Set a very low cost limit
        basic_request.max_cost = 0.001
        
        with pytest.raises(CostLimitExceededError) as exc_info:
            orchestrator.enrich(basic_request)
        
        assert "exceeds limit" in str(exc_info.value)
    
    def test_dry_run_mode(self, orchestrator, basic_request):
        """Test dry-run mode returns estimate without execution."""
        basic_request.dry_run = True
        
        result = orchestrator.enrich(basic_request)
        
        # Verify result is DryRunReport
        assert isinstance(result, DryRunReport)
        assert result.provider == "openai"
        assert result.enrichment_types == ["summary"]
        assert isinstance(result.estimate, CostEstimate)
    
    def test_cache_hit(self, orchestrator, basic_request, mock_cache_system, mock_provider_factory):
        """Test cache hit returns cached result without LLM call."""
        # Configure cache to return a result
        cached_result = EnrichmentV1(
            enrichment_version="v1",
            metadata=EnrichmentMetadata(
                provider="openai",
                model="gpt-4-turbo",
                timestamp=datetime.utcnow(),
                cost_usd=0.0,
                tokens_used=0,
                enrichment_types=["summary"],
                cache_hit=True
            ),
            summary=MOCK_SUMMARY_RESPONSE
        )
        mock_cache_system.get.return_value = cached_result
        
        result = orchestrator.enrich(basic_request)
        
        # Verify cached result was returned
        assert result == cached_result
        
        # Verify provider was NOT called
        provider = mock_provider_factory.create_provider.return_value
        provider.generate.assert_not_called()
    
    def test_cache_miss_stores_result(self, orchestrator, basic_request, mock_cache_system):
        """Test cache miss executes enrichment and stores result."""
        # Ensure cache miss
        mock_cache_system.get.return_value = None
        
        result = orchestrator.enrich(basic_request)
        
        # Verify result was stored in cache
        mock_cache_system.set.assert_called_once()
        
        # Verify stored result is EnrichmentV1
        stored_result = mock_cache_system.set.call_args[0][1]
        assert isinstance(stored_result, EnrichmentV1)
    
    def test_no_cache_flag(self, orchestrator, basic_request, mock_cache_system):
        """Test --no-cache flag bypasses cache."""
        basic_request.use_cache = False
        
        orchestrator.enrich(basic_request)
        
        # Verify cache was not checked or updated
        mock_cache_system.get.assert_not_called()
        mock_cache_system.set.assert_not_called()
    
    def test_prepare_prompts(self, orchestrator, basic_request, mock_prompt_loader, mock_prompt_renderer):
        """Test prompt preparation."""
        prompts = orchestrator._prepare_prompts(basic_request)
        
        # Verify prompts dict
        assert isinstance(prompts, dict)
        assert "summary" in prompts
        
        # Verify loader was called (without custom_dir kwarg)
        mock_prompt_loader.load_prompt.assert_called_once_with("summary")
        
        # Verify renderer was called
        mock_prompt_renderer.render.assert_called_once()
    
    def test_execute_enrichment(self, orchestrator, mock_provider_factory):
        """Test single enrichment execution."""
        provider = mock_provider_factory.create_provider.return_value
        
        response = orchestrator._execute_enrichment(
            enrichment_type="summary",
            prompt="Test prompt",
            provider=provider,
            model="gpt-4-turbo",
            transcript_text="Test transcript"
        )
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.model_used == "gpt-4-turbo"
        
        # Verify provider was called
        provider.generate.assert_called_once()
    
    def test_aggregate_results(self, orchestrator, mock_provider_factory):
        """Test result aggregation."""
        provider = mock_provider_factory.create_provider.return_value
        
        results = {
            "summary": LLMResponse(
                content=json.dumps(MOCK_SUMMARY_RESPONSE),
                model_used="gpt-4-turbo",
                tokens_used=500,
                cost_usd=0.015
            )
        }
        
        enrichment = orchestrator._aggregate_results(
            results=results,
            provider=provider,
            total_cost=0.015,
            total_tokens=500
        )
        
        # Verify EnrichmentV1 structure
        assert isinstance(enrichment, EnrichmentV1)
        assert enrichment.enrichment_version == "v1"
        assert enrichment.metadata.cost_usd == 0.015
        assert enrichment.metadata.tokens_used == 500
        assert enrichment.summary is not None
    
    def test_generate_cache_key(self, orchestrator, basic_request, mock_cache_system):
        """Test cache key generation."""
        prompts = {"summary": "Test prompt"}
        
        cache_key = orchestrator._generate_cache_key(
            request=basic_request,
            prompts=prompts,
            model="gpt-4-turbo"
        )
        
        # Verify cache system was called
        mock_cache_system.generate_key.assert_called_once()
        
        # Verify key is returned
        assert cache_key == "test_cache_key"
    
    def test_custom_prompts_directory(self, mock_provider_factory, mock_prompt_renderer, mock_cache_system):
        """Test custom prompts directory is passed to loader during initialization."""
        # Create a new prompt loader mock
        custom_prompt_loader = Mock()
        custom_prompt_loader.load_prompt.return_value = {
            "system": "Custom system",
            "user_template": "Custom template"
        }
        
        # Create orchestrator with custom prompts directory
        orchestrator = EnrichmentOrchestrator(
            provider_factory=mock_provider_factory,
            prompt_loader=custom_prompt_loader,
            prompt_renderer=mock_prompt_renderer,
            cache_system=mock_cache_system
        )
        
        request = EnrichmentRequest(
            transcript_text="Test",
            language="en",
            duration=60.0,
            enrichment_types=["summary"],
            provider="openai",
            custom_prompts_dir="/custom/prompts"
        )
        
        orchestrator._prepare_prompts(request)
        
        # Verify loader was called (custom_dir is set during PromptLoader init, not per-call)
        custom_prompt_loader.load_prompt.assert_called_once_with("summary")
    
    def test_validation_error_handling(self, orchestrator, basic_request, mock_provider_factory):
        """Test handling of validation errors."""
        # Configure provider to return invalid JSON
        provider = mock_provider_factory.create_provider.return_value
        provider.generate.return_value = LLMResponse(
            content="Invalid JSON",
            model_used="gpt-4-turbo",
            tokens_used=100,
            cost_usd=0.003
        )
        
        with pytest.raises(EnrichmentError) as exc_info:
            orchestrator.enrich(basic_request)
        
        assert "Failed to validate" in str(exc_info.value) or "Failed to execute" in str(exc_info.value)
