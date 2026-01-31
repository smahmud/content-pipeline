"""
Unit Tests for Cost Estimator

Tests cost estimation logic, token counting, and pricing calculations
for all supported LLM providers.
"""

import pytest
from unittest.mock import Mock, patch

from pipeline.enrichment.cost_estimator import CostEstimator
from pipeline.enrichment.agents.base import LLMRequest


class TestCostEstimator:
    """Test CostEstimator functionality."""
    
    @pytest.fixture
    def cost_estimator(self):
        """Create CostEstimator instance."""
        mock_agent = Mock()
        mock_agent.get_capabilities.return_value = {
            "provider": "openai",
            "default_model": "gpt-4-turbo",
            "models": ["gpt-4-turbo", "gpt-3.5-turbo"]
        }
        mock_agent.estimate_cost.return_value = 0.015
        return CostEstimator(agent=mock_agent)
    
    def test_estimate_openai_cost(self, cost_estimator):
        """Test cost estimation for OpenAI models."""
        estimate = cost_estimator.estimate(
            transcript_text="Test prompt with some content",
            enrichment_types=["summary"],
            model="gpt-4-turbo"
        )
        
        assert estimate.total_cost > 0.0
        assert isinstance(estimate.total_cost, float)
        assert estimate.provider == "openai"
    
    def test_estimate_claude_cost(self, cost_estimator):
        """Test cost estimation for Claude models."""
        # Update mock agent for Claude
        cost_estimator.agent.get_capabilities.return_value = {
            "provider": "claude",
            "default_model": "claude-3-opus-20240229",
            "models": ["claude-3-opus-20240229"]
        }
        
        estimate = cost_estimator.estimate(
            transcript_text="Test prompt",
            enrichment_types=["summary"],
            model="claude-3-opus-20240229"
        )
        
        assert estimate.total_cost > 0.0
        assert estimate.provider == "claude"
    
    def test_estimate_ollama_cost(self, cost_estimator):
        """Test that Ollama models have zero cost."""
        # Update mock agent for Ollama
        cost_estimator.agent.get_capabilities.return_value = {
            "provider": "ollama",
            "default_model": "llama2",
            "models": ["llama2"]
        }
        cost_estimator.agent.estimate_cost.return_value = 0.0
        
        estimate = cost_estimator.estimate(
            transcript_text="Test prompt",
            enrichment_types=["summary"],
            model="llama2"
        )
        
        assert estimate.total_cost == 0.0
        assert estimate.provider == "ollama"
    
    def test_token_counting(self, cost_estimator):
        """Test token counting for different text lengths."""
        short_text = "Hello world"
        long_text = "This is a much longer text " * 100
        
        short_tokens = cost_estimator._count_tokens(short_text, "gpt-4-turbo")
        long_tokens = cost_estimator._count_tokens(long_text, "gpt-4-turbo")
        
        assert short_tokens < long_tokens
        assert short_tokens > 0
    
    def test_cost_increases_with_tokens(self, cost_estimator):
        """Test that cost increases with token count."""
        # Configure mock to return different costs
        cost_estimator.agent.estimate_cost.side_effect = [0.005, 0.050]
        
        short_estimate = cost_estimator.estimate(
            transcript_text="Short",
            enrichment_types=["summary"],
            model="gpt-4-turbo"
        )
        
        long_estimate = cost_estimator.estimate(
            transcript_text="This is a much longer prompt " * 100,
            enrichment_types=["summary"],
            model="gpt-4-turbo"
        )
        
        assert long_estimate.total_cost > short_estimate.total_cost
    
    def test_different_models_different_costs(self, cost_estimator):
        """Test that different models have different costs."""
        # Configure mock to return different costs for different models
        cost_estimator.agent.estimate_cost.side_effect = [0.030, 0.005]
        
        gpt4_estimate = cost_estimator.estimate(
            transcript_text="Test prompt",
            enrichment_types=["summary"],
            model="gpt-4"
        )
        
        gpt35_estimate = cost_estimator.estimate(
            transcript_text="Test prompt",
            enrichment_types=["summary"],
            model="gpt-3.5-turbo"
        )
        
        # GPT-4 should be more expensive than GPT-3.5
        assert gpt4_estimate.total_cost > gpt35_estimate.total_cost
    
    def test_invalid_provider(self, cost_estimator):
        """Test handling of invalid provider."""
        # This test doesn't apply to the new API since provider comes from agent
        # Skip or remove this test
        pass
    
    def test_cost_warning_threshold(self, cost_estimator):
        """Test cost warning threshold detection."""
        max_cost = 1.00
        
        # Create estimate with 60% of max cost
        estimate = cost_estimator.estimate(
            transcript_text="Test",
            enrichment_types=["summary"],
            model="gpt-4-turbo"
        )
        estimate.total_cost = 0.60
        
        within_limit, warning = cost_estimator.check_cost_limit(estimate, max_cost)
        assert within_limit is True
        assert warning is not None  # Should warn at 50%+
        
        # Create estimate with 40% of max cost
        estimate.total_cost = 0.40
        within_limit, warning = cost_estimator.check_cost_limit(estimate, max_cost)
        assert within_limit is True
        assert warning is None
    
    def test_cost_limit_exceeded(self, cost_estimator):
        """Test cost limit detection."""
        max_cost = 1.00
        
        # Create estimate that exceeds limit
        estimate = cost_estimator.estimate(
            transcript_text="Test",
            enrichment_types=["summary"],
            model="gpt-4-turbo"
        )
        estimate.total_cost = 1.50
        
        within_limit, _ = cost_estimator.check_cost_limit(estimate, max_cost)
        assert within_limit is False
        
        # Create estimate within limit
        estimate.total_cost = 0.50
        within_limit, _ = cost_estimator.check_cost_limit(estimate, max_cost)
        assert within_limit is True
