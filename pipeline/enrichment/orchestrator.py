"""
Enrichment Orchestrator

Coordinates the complete enrichment workflow across all components.
Handles prompt loading, cost estimation, provider selection, execution,
caching, and result aggregation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse
from pipeline.llm.factory import LLMProviderFactory
from pipeline.enrichment.cost_estimator import CostEstimator, CostEstimate
from pipeline.enrichment.prompts.loader import PromptLoader
from pipeline.enrichment.prompts.renderer import PromptRenderer
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1, EnrichmentMetadata
from pipeline.enrichment.validate import validate_and_repair_enrichment
from pipeline.enrichment.cache import CacheSystem
from pipeline.enrichment.errors import (
    CostLimitExceededError,
    EnrichmentError,
    SchemaValidationError
)


@dataclass
class EnrichmentRequest:
    """Request for enrichment operation.
    
    Attributes:
        transcript_text: The transcript text to enrich
        language: Language code (e.g., "en")
        duration: Duration in seconds
        enrichment_types: List of enrichment types to generate
        provider: LLM provider to use
        model: Optional specific model
        max_cost: Optional cost limit in USD
        dry_run: If True, only estimate costs without execution
        use_cache: Whether to use caching
        custom_prompts_dir: Optional custom prompts directory
    """
    transcript_text: str
    language: str
    duration: float
    enrichment_types: List[str]
    provider: str
    model: Optional[str] = None
    max_cost: Optional[float] = None
    dry_run: bool = False
    use_cache: bool = True
    custom_prompts_dir: Optional[str] = None


@dataclass
class DryRunReport:
    """Report for dry-run mode.
    
    Attributes:
        estimate: Cost estimate
        enrichment_types: List of enrichment types
        provider: Provider name
        model: Model name
        would_use_cache: Whether cache would be used
        cache_hit: Whether result is in cache
    """
    estimate: CostEstimate
    enrichment_types: List[str]
    provider: str
    model: str
    would_use_cache: bool
    cache_hit: bool


class EnrichmentOrchestrator:
    """Orchestrates enrichment workflow across all components.
    
    This class coordinates:
    - Prompt loading and rendering
    - Cost estimation and limit enforcement
    - Provider selection and execution
    - Result validation and aggregation
    - Caching (when implemented)
    
    Example:
        >>> orchestrator = EnrichmentOrchestrator(provider_factory)
        >>> request = EnrichmentRequest(...)
        >>> result = orchestrator.enrich(request)
    """
    
    def __init__(
        self,
        provider_factory: LLMProviderFactory,
        prompt_loader: Optional[PromptLoader] = None,
        prompt_renderer: Optional[PromptRenderer] = None,
        cache_system: Optional[CacheSystem] = None
    ):
        """Initialize orchestrator.
        
        Args:
            provider_factory: Factory for creating LLM providers
            prompt_loader: Optional prompt loader (creates default if not provided)
            prompt_renderer: Optional prompt renderer (creates default if not provided)
            cache_system: Optional cache system (creates default if not provided)
        """
        self.provider_factory = provider_factory
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_renderer = prompt_renderer or PromptRenderer()
        self.cache_system = cache_system or CacheSystem()
    
    def enrich(self, request: EnrichmentRequest) -> EnrichmentV1 | DryRunReport:
        """Execute enrichment workflow.
        
        Args:
            request: Enrichment request with all parameters
            
        Returns:
            EnrichmentV1 result or DryRunReport if dry_run=True
            
        Raises:
            CostLimitExceededError: If estimated cost exceeds max_cost
            EnrichmentError: If enrichment fails
        """
        # 1. Create provider
        provider = self.provider_factory.create_provider(request.provider)
        
        # 2. Load and render prompts
        prompts = self._prepare_prompts(request)
        
        # 3. Estimate costs
        cost_estimator = CostEstimator(provider)
        estimate = cost_estimator.estimate(
            transcript_text=request.transcript_text,
            enrichment_types=request.enrichment_types,
            model=request.model
        )
        
        # 4. Check cost limit
        if request.max_cost:
            within_limit, warning = cost_estimator.check_cost_limit(
                estimate, request.max_cost
            )
            
            if not within_limit:
                raise CostLimitExceededError(
                    f"Estimated cost ${estimate.total_cost:.4f} exceeds "
                    f"limit ${request.max_cost:.4f}"
                )
            
            if warning:
                # In a real implementation, this would be logged or displayed
                # For now, we just note it in the metadata
                pass
        
        # 5. Dry-run mode
        if request.dry_run:
            # Check if result would be in cache
            cache_hit = False
            if request.use_cache:
                cache_key = self._generate_cache_key(request, prompts, estimate.model)
                cached_result = self.cache_system.get(cache_key)
                cache_hit = cached_result is not None
            
            return DryRunReport(
                estimate=estimate,
                enrichment_types=request.enrichment_types,
                provider=request.provider,
                model=estimate.model,
                would_use_cache=request.use_cache,
                cache_hit=cache_hit
            )
        
        # 6. Check cache
        if request.use_cache:
            cache_key = self._generate_cache_key(request, prompts, estimate.model)
            cached_result = self.cache_system.get(cache_key)
            
            if cached_result is not None:
                # Cache hit! Return cached result
                return cached_result
        
        # 7. Execute enrichment for each type
        results = {}
        total_cost = 0.0
        total_tokens = 0
        
        for enrichment_type in request.enrichment_types:
            prompt = prompts[enrichment_type]
            
            # DEBUG: Log enrichment execution
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Executing enrichment: {enrichment_type}",
                extra={
                    "enrichment_type": enrichment_type,
                    "prompt_length": len(prompt),
                    "multi_enrichment_mode": len(request.enrichment_types) > 1,
                    "enrichment_order": request.enrichment_types,
                    "current_index": request.enrichment_types.index(enrichment_type)
                }
            )
            
            result = self._execute_enrichment(
                enrichment_type=enrichment_type,
                prompt=prompt,
                provider=provider,
                model=request.model,
                transcript_text=request.transcript_text
            )
            
            # DEBUG: Log result type
            logger.debug(
                f"Enrichment result for {enrichment_type}",
                extra={
                    "enrichment_type": enrichment_type,
                    "content_length": len(result.content),
                    "content_preview": result.content[:200],
                    "tokens_used": result.tokens_used,
                    "cost_usd": result.cost_usd
                }
            )
            
            results[enrichment_type] = result
            total_cost += result.cost_usd
            total_tokens += result.tokens_used
        
        # 8. Aggregate results into EnrichmentV1
        enrichment = self._aggregate_results(
            results=results,
            provider=provider,
            total_cost=total_cost,
            total_tokens=total_tokens
        )
        
        # 9. Cache result
        if request.use_cache:
            cache_key = self._generate_cache_key(request, prompts, estimate.model)
            self.cache_system.set(cache_key, enrichment)
        
        return enrichment
    
    def _prepare_prompts(self, request: EnrichmentRequest) -> Dict[str, str]:
        """Load and render prompts for all enrichment types.
        
        Args:
            request: Enrichment request
            
        Returns:
            Dict mapping enrichment type to rendered prompt
        """
        prompts = {}
        
        # Prepare additional context for rendering
        word_count = len(request.transcript_text.split())
        additional_context = {
            "word_count": word_count
        }
        
        for enrichment_type in request.enrichment_types:
            # Load template (custom_prompts_dir is already set in PromptLoader init)
            template = self.prompt_loader.load_prompt(enrichment_type)
            
            # Render with proper arguments
            prompt = self.prompt_renderer.render(
                prompt_template=template,
                transcript_text=request.transcript_text,
                transcript_language=request.language,
                transcript_duration=str(request.duration),
                additional_context=additional_context
            )
            prompts[enrichment_type] = prompt
        
        return prompts
    
    def _execute_enrichment(
        self,
        enrichment_type: str,
        prompt: str,
        provider: BaseLLMProvider,
        model: Optional[str],
        transcript_text: str
    ) -> LLMResponse:
        """Execute enrichment for a single type.
        
        Args:
            enrichment_type: Type of enrichment
            prompt: Rendered prompt
            provider: LLM provider to use
            model: Optional specific model
            transcript_text: Original transcript text
            
        Returns:
            LLM response
            
        Raises:
            EnrichmentError: If enrichment fails
        """
        # Determine max tokens based on enrichment type
        max_tokens_map = {
            "summary": 1000,
            "tag": 500,
            "chapter": 2000,
            "highlight": 1500
        }
        max_tokens = max_tokens_map.get(enrichment_type, 1000)
        
        # Create request
        llm_request = LLMRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            model=model
        )
        
        # Note: Current implementation assumes transcript fits in context window.
        # For very long transcripts, chunking may be needed in future versions.
        
        try:
            # Execute LLM call
            response = provider.generate(llm_request)
            
            # Validate and repair response
            validated_content = validate_and_repair_enrichment(
                response.content,
                enrichment_type
            )
            
            # Update response with validated content
            response.content = validated_content
            
            return response
        
        except SchemaValidationError as e:
            raise EnrichmentError(
                f"Failed to validate {enrichment_type} enrichment: {e}"
            )
        except Exception as e:
            raise EnrichmentError(
                f"Failed to execute {enrichment_type} enrichment: {e}"
            )
    
    def _aggregate_results(
        self,
        results: Dict[str, LLMResponse],
        provider: BaseLLMProvider,
        total_cost: float,
        total_tokens: int
    ) -> EnrichmentV1:
        """Aggregate enrichment results into EnrichmentV1 schema.
        
        Args:
            results: Dict mapping enrichment type to LLM response
            provider: LLM provider used
            total_cost: Total cost across all enrichments
            total_tokens: Total tokens used
            
        Returns:
            EnrichmentV1 container
        """
        capabilities = provider.get_capabilities()
        
        # Create metadata
        metadata = EnrichmentMetadata(
            provider=capabilities.get("provider", "unknown"),
            model=results[list(results.keys())[0]].model_used,
            timestamp=datetime.utcnow(),
            cost_usd=total_cost,
            tokens_used=total_tokens,
            enrichment_types=list(results.keys())
        )
        
        # Parse enrichment results
        enrichments = {}
        for enrichment_type, response in results.items():
            try:
                # Parse JSON from response content
                enrichment_data = json.loads(response.content)
                enrichments[enrichment_type] = enrichment_data
            except json.JSONDecodeError:
                # If not valid JSON, wrap in error structure
                enrichments[enrichment_type] = {
                    "error": "Failed to parse enrichment response",
                    "raw_content": response.content
                }
        
        # Create EnrichmentV1 container with proper field names
        return EnrichmentV1(
            enrichment_version="v1",
            metadata=metadata,
            summary=enrichments.get("summary"),
            tags=enrichments.get("tag"),
            chapters=enrichments.get("chapter"),
            highlights=enrichments.get("highlight")
        )
    
    def _generate_cache_key(
        self,
        request: EnrichmentRequest,
        prompts: Dict[str, str],
        model: str
    ) -> str:
        """Generate cache key for the enrichment request.
        
        Args:
            request: Enrichment request
            prompts: Rendered prompts for each enrichment type
            model: Model being used
            
        Returns:
            Cache key (hash)
        """
        # Combine all prompts into a single template string
        combined_prompts = "\n---\n".join(
            f"{etype}:\n{prompt}"
            for etype, prompt in sorted(prompts.items())
        )
        
        # Generate cache key
        return self.cache_system.generate_key(
            transcript_text=request.transcript_text,
            model=model,
            prompt_template=combined_prompts,
            enrichment_types=sorted(request.enrichment_types),
            parameters={
                "language": request.language,
                "duration": request.duration
            }
        )
