"""
Cost Estimator

Provides pre-flight cost estimation for enrichment operations.
Calculates expected costs based on transcript length, model pricing,
and enrichment types requested.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import tiktoken

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest


@dataclass
class CostEstimate:
    """Cost estimation result.
    
    Attributes:
        total_cost: Total estimated cost in USD
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        input_tokens: Estimated input token count
        output_tokens: Estimated output token count
        model: Model used for estimation
        provider: Provider name
        breakdown: Per-enrichment-type cost breakdown
    """
    total_cost: float
    input_cost: float
    output_cost: float
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    breakdown: Dict[str, float]


class CostEstimator:
    """Estimates costs for enrichment operations before execution.
    
    This class provides pre-flight cost estimation to help users make
    informed decisions about whether to proceed with enrichment operations.
    It considers transcript length, model pricing, and enrichment types.
    """
    
    # Expected output tokens per enrichment type (conservative estimates)
    OUTPUT_TOKEN_ESTIMATES = {
        "summary": 500,      # Short + medium + long summaries
        "tag": 200,          # Categories, keywords, entities
        "chapter": 1000,     # Multiple chapters with descriptions
        "highlight": 800,    # Multiple highlights with context
    }
    
    # Prompt overhead tokens (system prompt + formatting)
    PROMPT_OVERHEAD = 300
    
    def __init__(self, agent: BaseLLMAgent):
        """Initialize cost estimator.
        
        Args:
            agent: LLM agent to use for cost estimation
        """
        self.agent = agent
        self._tokenizer_cache: Dict[str, tiktoken.Encoding] = {}
    
    def estimate(
        self,
        transcript_text: str,
        enrichment_types: List[str],
        model: Optional[str] = None,
        prompt_templates: Optional[Dict[str, str]] = None
    ) -> CostEstimate:
        """Estimate cost for enrichment operation.
        
        Args:
            transcript_text: The transcript text to enrich
            enrichment_types: List of enrichment types to generate
            model: Optional specific model to use
            prompt_templates: Optional rendered prompt templates per type
            
        Returns:
            Cost estimation with breakdown
        """
        # Get model info
        capabilities = self.agent.get_capabilities()
        model_used = model or capabilities.get("default_model", "")
        provider = capabilities.get("provider", "unknown")
        
        # Count input tokens
        input_tokens = self._count_tokens(transcript_text, model_used)
        input_tokens += self.PROMPT_OVERHEAD  # Add prompt overhead
        
        # Estimate output tokens
        output_tokens = sum(
            self.OUTPUT_TOKEN_ESTIMATES.get(etype, 500)
            for etype in enrichment_types
        )
        
        # Calculate costs per enrichment type
        breakdown = {}
        total_input_cost = 0.0
        total_output_cost = 0.0
        
        for etype in enrichment_types:
            # Create a request for this enrichment type
            etype_output_tokens = self.OUTPUT_TOKEN_ESTIMATES.get(etype, 500)
            
            request = LLMRequest(
                prompt=transcript_text,
                max_tokens=etype_output_tokens,
                temperature=0.3,
                model=model_used
            )
            
            # Get cost estimate from agent
            etype_cost = self.agent.estimate_cost(request)
            breakdown[etype] = etype_cost
            
            # Accumulate costs (rough approximation)
            # In reality, each enrichment type has different input/output ratios
            total_input_cost += etype_cost * 0.3  # Assume 30% input cost
            total_output_cost += etype_cost * 0.7  # Assume 70% output cost
        
        total_cost = sum(breakdown.values())
        
        return CostEstimate(
            total_cost=total_cost,
            input_cost=total_input_cost,
            output_cost=total_output_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_used,
            provider=provider,
            breakdown=breakdown
        )
    
    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for the specified model.
        
        Args:
            text: Text to count tokens for
            model: Model identifier
            
        Returns:
            Token count
        """
        # Try to get tokenizer for model
        try:
            encoding = self._get_tokenizer(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough word-based estimation
            # Average English word is ~1.3 tokens
            return int(len(text.split()) * 1.3)
    
    def _get_tokenizer(self, model: str) -> tiktoken.Encoding:
        """Get tokenizer for model (with caching).
        
        Args:
            model: Model identifier
            
        Returns:
            Tokenizer encoding
        """
        if model in self._tokenizer_cache:
            return self._tokenizer_cache[model]
        
        # Try to get encoding for model
        try:
            # OpenAI models
            if "gpt" in model.lower():
                encoding = tiktoken.encoding_for_model(model)
            # Claude models (use cl100k_base as approximation)
            elif "claude" in model.lower():
                encoding = tiktoken.get_encoding("cl100k_base")
            # Fallback to cl100k_base
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            self._tokenizer_cache[model] = encoding
            return encoding
        
        except Exception:
            # If tiktoken fails, use cl100k_base as fallback
            encoding = tiktoken.get_encoding("cl100k_base")
            self._tokenizer_cache[model] = encoding
            return encoding
    
    def check_cost_limit(
        self,
        estimate: CostEstimate,
        max_cost: float,
        warning_threshold: float = 0.5
    ) -> tuple[bool, Optional[str]]:
        """Check if estimated cost is within limits.
        
        Args:
            estimate: Cost estimate to check
            max_cost: Maximum allowed cost in USD
            warning_threshold: Threshold for warning (0.5 = 50%)
            
        Returns:
            Tuple of (within_limit, warning_message)
            - within_limit: True if cost is within max_cost
            - warning_message: Warning message if cost exceeds threshold, None otherwise
        """
        # Check if cost exceeds limit
        if estimate.total_cost > max_cost:
            return False, None
        
        # Check if cost exceeds warning threshold
        warning_cost = max_cost * warning_threshold
        if estimate.total_cost > warning_cost:
            percentage = (estimate.total_cost / max_cost) * 100
            warning = (
                f"⚠️  Cost estimate (${estimate.total_cost:.4f}) is {percentage:.1f}% "
                f"of your max cost limit (${max_cost:.4f})"
            )
            return True, warning
        
        return True, None
    
    def format_estimate(self, estimate: CostEstimate) -> str:
        """Format cost estimate for display.
        
        Args:
            estimate: Cost estimate to format
            
        Returns:
            Formatted string
        """
        lines = [
            "Cost Estimate:",
            f"  Provider: {estimate.provider}",
            f"  Model: {estimate.model}",
            f"  Input tokens: {estimate.input_tokens:,}",
            f"  Output tokens: {estimate.output_tokens:,}",
            f"  Total cost: ${estimate.total_cost:.4f}",
            "",
            "Breakdown by enrichment type:"
        ]
        
        for etype, cost in estimate.breakdown.items():
            lines.append(f"  - {etype}: ${cost:.4f}")
        
        return "\n".join(lines)
