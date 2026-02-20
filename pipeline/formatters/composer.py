"""
FormatComposer - Orchestrates the complete formatting workflow.

This module provides the main orchestrator that coordinates generators,
template engine, LLM enhancer, and platform validator to produce
formatted output from enriched content.

Implements:
- Single format generation (format_single)
- Bundle generation (format_bundle) - named bundle configurations
- Batch processing (format_batch) - multiple input files
- Cost estimation and control
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.formatters.base import (
    FormatRequest,
    FormatResult,
    CostEstimate,
    VALID_OUTPUT_TYPES,
)
from pipeline.formatters.bundles.loader import BundleLoader, BundleConfig, BundleNotFoundError
from pipeline.formatters.errors import (
    FormatError,
    InputValidationError,
    EnhancementError,
    CostLimitExceededError,
)
from pipeline.formatters.generators.factory import GeneratorFactory, register_all_generators
from pipeline.formatters.input_validator import InputValidator
from pipeline.formatters.llm.enhancer import LLMEnhancer, EnhancementResult
from pipeline.formatters.schemas.format_v1 import FormatV1, LLMMetadata, ValidationMetadata
from pipeline.formatters.style_profile import StyleProfile, StyleProfileLoader
from pipeline.formatters.template_engine import TemplateEngine
from pipeline.formatters.validator import PlatformValidator
from pipeline.formatters.source_combiner import SourceCombiner
from pipeline.formatters.image_prompts import ImagePromptGenerator, ImagePromptsResult
from pipeline.formatters.code_samples import CodeSampleGenerator, CodeSamplesResult


logger = logging.getLogger(__name__)


# Cost warning threshold (50% of max_cost)
COST_WARNING_THRESHOLD = 0.5


@dataclass
class BundleResult:
    """Result of bundle generation (--bundle flag).
    
    Attributes:
        bundle_name: Name of the bundle that was generated
        successful: List of output types successfully generated
        failed: List of (output_type, error_message) tuples
        output_dir: Directory where outputs were written
        manifest_path: Path to manifest file
        total_cost: Total LLM cost in USD
        total_time: Total processing time in seconds
        results: Dictionary mapping output type to FormatResult
    """
    bundle_name: str = ""
    successful: List[str] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)
    output_dir: str = ""
    manifest_path: str = ""
    total_cost: float = 0.0
    total_time: float = 0.0
    results: Dict[str, FormatResult] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing.
    
    Attributes:
        successful: List of file paths successfully processed
        failed: List of (file_path, error_message) tuples
        output_dir: Directory where outputs were written
        total_cost: Total LLM cost in USD
        total_time: Total processing time in seconds
        results: Dictionary mapping file path to FormatResult
    """
    successful: List[str] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)
    output_dir: str = ""
    total_cost: float = 0.0
    total_time: float = 0.0
    results: Dict[str, FormatResult] = field(default_factory=dict)


class FormatComposer:
    """Orchestrates the complete formatting workflow.
    
    The FormatComposer coordinates all formatting components:
    - GeneratorFactory: Creates output type generators
    - TemplateEngine: Renders Jinja2 templates
    - LLMEnhancer: Enhances prose with LLM (optional)
    - PlatformValidator: Validates and truncates for platform limits
    
    Example:
        >>> composer = FormatComposer()
        >>> request = FormatRequest(
        ...     enriched_content=enriched_data,
        ...     output_type="blog",
        ...     llm_enhance=True
        ... )
        >>> result = composer.format_single(request)
    """
    
    def __init__(
        self,
        generator_factory: Optional[GeneratorFactory] = None,
        template_engine: Optional[TemplateEngine] = None,
        llm_enhancer: Optional[LLMEnhancer] = None,
        platform_validator: Optional[PlatformValidator] = None,
        style_profile_loader: Optional[StyleProfileLoader] = None,
        input_validator: Optional[InputValidator] = None,
        bundle_loader: Optional[BundleLoader] = None,
        source_combiner: Optional[SourceCombiner] = None,
        image_prompt_generator: Optional[ImagePromptGenerator] = None,
        code_sample_generator: Optional[CodeSampleGenerator] = None,
    ):
        """Initialize the FormatComposer.
        
        Args:
            generator_factory: Factory for creating generators (created if None)
            template_engine: Template engine instance (created if None)
            llm_enhancer: LLM enhancer instance (created if None)
            platform_validator: Platform validator instance (created if None)
            style_profile_loader: Style profile loader (created if None)
            input_validator: Input validator (created if None)
            bundle_loader: Bundle loader instance (created if None)
            source_combiner: SourceCombiner for multi-source input (created if None)
            image_prompt_generator: ImagePromptGenerator for image prompts (optional)
            code_sample_generator: CodeSampleGenerator for code samples (optional)
        """
        # Initialize components
        self._template_engine = template_engine or TemplateEngine()
        self._platform_validator = platform_validator or PlatformValidator()
        
        # Generator factory needs template engine and validator
        if generator_factory is None:
            register_all_generators()
            self._generator_factory = GeneratorFactory(
                template_engine=self._template_engine,
                platform_validator=self._platform_validator,
            )
        else:
            self._generator_factory = generator_factory
        
        self._llm_enhancer = llm_enhancer
        self._style_profile_loader = style_profile_loader or StyleProfileLoader()
        self._input_validator = input_validator or InputValidator()
        self._bundle_loader = bundle_loader or BundleLoader()
        self._source_combiner = source_combiner or SourceCombiner()
        self._image_prompt_generator = image_prompt_generator
        self._code_sample_generator = code_sample_generator

    def format_single(self, request: FormatRequest) -> FormatResult:
        """Format a single output type from enriched content.
        
        This is the main entry point for single format generation.
        It coordinates validation, generation, enhancement, and platform validation.
        
        Args:
            request: FormatRequest with enriched content and options
            
        Returns:
            FormatResult with formatted content and metadata
            
        Raises:
            InputValidationError: If input validation fails
            CostLimitExceededError: If estimated cost exceeds max_cost
            FormatError: If formatting fails
        """
        warnings: List[str] = []
        start_time = datetime.now()
        
        # Step 1: Validate input
        validation_result = self._input_validator.validate_content(
            request.enriched_content,
            request.output_type
        )
        if not validation_result.is_valid:
            raise InputValidationError(
                message=f"Input validation failed: {'; '.join(validation_result.errors)}",
                field_name=None,
                expected="Valid EnrichmentV1 content with required fields",
                actual="Missing or invalid fields",
            )
        
        # Step 2: Load style profile if specified
        style_profile = None
        if request.style_profile:
            if isinstance(request.style_profile, dict):
                # Already parsed
                style_profile = StyleProfile(**request.style_profile)
            elif isinstance(request.style_profile, StyleProfile):
                style_profile = request.style_profile
        
        # Step 3: Cost estimation (if LLM enhancement enabled and max_cost set)
        llm_metadata = None
        if request.llm_enhance and not request.dry_run:
            if request.max_cost is not None:
                cost_estimate = self.estimate_cost(request)
                if not cost_estimate.within_budget:
                    raise CostLimitExceededError(
                        message=f"Estimated cost ${cost_estimate.estimated_cost_usd:.4f} exceeds limit ${request.max_cost:.4f}",
                        estimated_cost=cost_estimate.estimated_cost_usd,
                        max_cost=request.max_cost,
                        output_type=request.output_type,
                    )
                # Check warning threshold
                if cost_estimate.estimated_cost_usd > request.max_cost * COST_WARNING_THRESHOLD:
                    warnings.append(
                        f"Cost estimate (${cost_estimate.estimated_cost_usd:.4f}) is "
                        f"{(cost_estimate.estimated_cost_usd / request.max_cost * 100):.1f}% of max cost"
                    )
        
        # Step 4: Dry run - return estimate only
        if request.dry_run:
            cost_estimate = self.estimate_cost(request)
            return FormatResult(
                content="",
                metadata=self._create_metadata(
                    request=request,
                    source_file="",
                    style_profile_name=style_profile.name if style_profile else None,
                    llm_metadata=None,
                    validation_metadata=ValidationMetadata(
                        platform=request.platform,
                        character_count=0,
                        truncated=False,
                        warnings=[]
                    ),
                ),
                warnings=[f"Dry run - estimated cost: ${cost_estimate.estimated_cost_usd:.4f}"],
                success=True,
            )
        
        # Step 5: Get generator and generate base content
        generator = self._generator_factory.get_generator(request.output_type)
        base_result = generator.format(request)
        
        if not base_result.success:
            return base_result
        
        content = base_result.content
        warnings.extend(base_result.warnings)
        
        # Step 6: LLM enhancement (if enabled)
        if request.llm_enhance and self._llm_enhancer:
            enhancement_result = self._llm_enhancer.enhance(
                content=content,
                output_type=request.output_type,
                style_profile=style_profile,
                tone=request.tone,
                length=request.length,
                provider=request.provider,
                model=request.model,
                url=request.url,
            )
            
            content = enhancement_result.content
            warnings.extend(enhancement_result.warnings)
            
            if enhancement_result.enhanced:
                llm_metadata = LLMMetadata(
                    provider=enhancement_result.provider,
                    model=enhancement_result.model,
                    cost_usd=enhancement_result.cost_usd,
                    tokens_used=enhancement_result.tokens_used,
                    temperature=0.7,  # Default, could be from config
                    enhanced=True,
                )
            elif not enhancement_result.success:
                # Graceful degradation - content is template-only
                llm_metadata = LLMMetadata(
                    provider=enhancement_result.provider or "none",
                    model=enhancement_result.model or "none",
                    cost_usd=0.0,
                    tokens_used=0,
                    temperature=0.0,
                    enhanced=False,
                )
        
        # Step 7: Platform validation (if platform specified)
        validation_metadata = ValidationMetadata(
            platform=request.platform,
            character_count=len(content),
            truncated=False,
            warnings=[],
        )
        
        if request.platform:
            content, validation_result = self._platform_validator.validate_and_truncate(
                content=content,
                platform=request.platform,
                auto_truncate=True,
            )
            validation_metadata = ValidationMetadata(
                platform=request.platform,
                character_count=validation_result.character_count,
                truncated=validation_result.truncated,
                warnings=validation_result.warnings,
            )
            warnings.extend(validation_result.warnings)
        
        # Step 8: Create final result
        metadata = self._create_metadata(
            request=request,
            source_file=request.enriched_content.get("source_file", ""),
            style_profile_name=style_profile.name if style_profile else None,
            llm_metadata=llm_metadata,
            validation_metadata=validation_metadata,
        )
        
        return FormatResult(
            content=content,
            metadata=metadata,
            warnings=warnings,
            success=True,
        )

    def format_from_sources(
        self,
        sources_folder: Path,
        output_type: str,
        platform: Optional[str] = None,
        style_profile: Optional[dict] = None,
        tone: Optional[str] = None,
        length: Optional[str] = None,
        llm_enhance: bool = True,
        provider: str = "auto",
        model: Optional[str] = None,
        max_cost: Optional[float] = None,
        dry_run: bool = False,
        url: Optional[str] = None,
    ) -> FormatResult:
        """Format content from multiple source files in a folder.

        Loads all supported files from the folder, combines them into
        unified enriched content, then formats using the standard pipeline.

        Args:
            sources_folder: Path to folder containing source files
            output_type: Target output format
            platform: Optional target platform
            style_profile: Optional style profile dict
            tone: Optional tone override
            length: Optional length override
            llm_enhance: Whether to enable LLM enhancement
            provider: LLM provider
            model: Optional specific model
            max_cost: Optional cost limit
            dry_run: If True, estimate cost only
            url: Optional URL for promotional content

        Returns:
            FormatResult with formatted content
        """
        logger.info(f"Loading sources from {sources_folder}")

        # Load and combine sources
        try:
            sources = self._source_combiner.load_sources(sources_folder)
        except (FileNotFoundError, ValueError) as exc:
            return FormatResult(
                content="",
                metadata=self._create_metadata(
                    request=FormatRequest(
                        enriched_content={},
                        output_type=output_type,
                    ),
                    source_file=str(sources_folder),
                    style_profile_name=None,
                    llm_metadata=None,
                    validation_metadata=ValidationMetadata(
                        platform=platform,
                        character_count=0,
                        truncated=False,
                        warnings=[],
                    ),
                ),
                warnings=[str(exc)],
                success=False,
                error=str(exc),
            )

        if not sources:
            return FormatResult(
                content="",
                metadata=self._create_metadata(
                    request=FormatRequest(
                        enriched_content={},
                        output_type=output_type,
                    ),
                    source_file=str(sources_folder),
                    style_profile_name=None,
                    llm_metadata=None,
                    validation_metadata=ValidationMetadata(
                        platform=platform,
                        character_count=0,
                        truncated=False,
                        warnings=[],
                    ),
                ),
                warnings=[f"No supported files found in {sources_folder}"],
                success=False,
                error=f"No supported files found in {sources_folder}",
            )

        combined = self._source_combiner.combine(sources)
        logger.info(
            f"Combined {combined.source_count} sources "
            f"({len(combined.warnings)} warnings)"
        )

        # Create format request with combined content
        request = FormatRequest(
            enriched_content=combined.enriched_content,
            output_type=output_type,
            platform=platform,
            style_profile=style_profile,
            tone=tone,
            length=length,
            llm_enhance=llm_enhance,
            provider=provider,
            model=model,
            max_cost=max_cost,
            dry_run=dry_run,
            url=url,
        )

        # Format using standard pipeline
        result = self.format_single(request)

        # Append source combiner warnings
        if combined.warnings:
            result.warnings.extend(combined.warnings)

        return result

    def generate_image_prompts(
        self,
        enriched_content: dict,
        output_type: str,
        platform: Optional[str] = None,
    ) -> Optional[ImagePromptsResult]:
        """Generate image prompts for content.

        Args:
            enriched_content: The enriched content data
            output_type: Target output format
            platform: Optional target platform

        Returns:
            ImagePromptsResult or None if not supported/available
        """
        if self._image_prompt_generator is None:
            self._image_prompt_generator = ImagePromptGenerator()

        if not self._image_prompt_generator.is_supported(output_type):
            logger.info(
                f"Image prompts not supported for output type: {output_type}"
            )
            return None

        return self._image_prompt_generator.generate(
            enriched_content=enriched_content,
            output_type=output_type,
            platform=platform,
        )

    def generate_code_samples(
        self,
        enriched_content: dict,
        output_type: str,
    ) -> Optional[CodeSamplesResult]:
        """Generate code samples for technical content.

        Args:
            enriched_content: The enriched content data
            output_type: Target output format

        Returns:
            CodeSamplesResult or None if not supported/technical
        """
        if self._code_sample_generator is None:
            self._code_sample_generator = CodeSampleGenerator()

        if not self._code_sample_generator.is_supported(output_type):
            logger.info(
                f"Code samples not supported for output type: {output_type}"
            )
            return None

        if not self._code_sample_generator.is_technical_content(enriched_content):
            logger.info("Content is not technical, skipping code samples")
            return None

        return self._code_sample_generator.generate(
            enriched_content=enriched_content,
            output_type=output_type,
        )

    @property
    def source_combiner(self) -> SourceCombiner:
        """Get the source combiner instance."""
        return self._source_combiner

    @property
    def image_prompt_generator(self) -> Optional[ImagePromptGenerator]:
        """Get the image prompt generator instance."""
        return self._image_prompt_generator

    @property
    def code_sample_generator(self) -> Optional[CodeSampleGenerator]:
        """Get the code sample generator instance."""
        return self._code_sample_generator

    def estimate_cost(self, request: FormatRequest) -> CostEstimate:
        """Estimate LLM enhancement cost before execution.
        
        Implements Property 9: Cost Estimation Before Execution.
        
        Args:
            request: FormatRequest to estimate cost for
            
        Returns:
            CostEstimate with estimated tokens and cost
        """
        if not self._llm_enhancer:
            return CostEstimate(
                estimated_tokens=0,
                estimated_cost_usd=0.0,
                provider="none",
                model="none",
                within_budget=True,
            )
        
        # Get a sample of the content to estimate
        # We need to generate template content first to estimate enhancement cost
        try:
            generator = self._generator_factory.get_generator(request.output_type)
            
            # Create a non-enhancing request for template generation
            template_request = FormatRequest(
                enriched_content=request.enriched_content,
                output_type=request.output_type,
                platform=request.platform,
                style_profile=request.style_profile,
                tone=request.tone,
                length=request.length,
                llm_enhance=False,  # Don't enhance for estimation
            )
            
            template_result = generator.format(template_request)
            content = template_result.content
            
        except Exception as e:
            logger.warning(f"Failed to generate template for cost estimation: {e}")
            # Use a rough estimate based on typical content size
            content = "x" * 2000  # Assume ~2000 chars
        
        # Estimate cost using LLM enhancer
        estimated_cost = self._llm_enhancer.estimate_cost(
            content=content,
            output_type=request.output_type,
            provider=request.provider,
            model=request.model,
        )
        
        # Rough token estimate (4 chars per token average)
        estimated_tokens = len(content) // 4 + 500  # Add output tokens
        
        # Check if within budget
        within_budget = True
        if request.max_cost is not None:
            within_budget = estimated_cost <= request.max_cost
        
        return CostEstimate(
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            provider=request.provider,
            model=request.model or "default",
            within_budget=within_budget,
        )
    
    def _create_metadata(
        self,
        request: FormatRequest,
        source_file: str,
        style_profile_name: Optional[str],
        llm_metadata: Optional[LLMMetadata],
        validation_metadata: ValidationMetadata,
    ) -> FormatV1:
        """Create FormatV1 metadata for the result.
        
        Args:
            request: Original format request
            source_file: Source file path
            style_profile_name: Name of style profile used
            llm_metadata: LLM enhancement metadata
            validation_metadata: Platform validation metadata
            
        Returns:
            FormatV1 metadata object
        """
        return FormatV1(
            format_version="v1",
            output_type=request.output_type,
            platform=request.platform,
            timestamp=datetime.now(),
            source_file=source_file,
            style_profile_used=style_profile_name,
            llm_metadata=llm_metadata,
            validation=validation_metadata,
            tone=request.tone,
            length=request.length,
        )
    
    def get_available_output_types(self) -> List[str]:
        """Get list of available output types.
        
        Returns:
            List of output type strings
        """
        return self._generator_factory.get_registered_types()
    
    def get_generator(self, output_type: str):
        """Get a generator for the specified output type.
        
        Args:
            output_type: Output type to get generator for
            
        Returns:
            Generator instance
        """
        return self._generator_factory.get_generator(output_type)
    
    @property
    def llm_enhancer(self) -> Optional[LLMEnhancer]:
        """Get the LLM enhancer instance."""
        return self._llm_enhancer
    
    @llm_enhancer.setter
    def llm_enhancer(self, enhancer: Optional[LLMEnhancer]) -> None:
        """Set the LLM enhancer instance."""
        self._llm_enhancer = enhancer
    
    @property
    def template_engine(self) -> TemplateEngine:
        """Get the template engine instance."""
        return self._template_engine
    
    @property
    def platform_validator(self) -> PlatformValidator:
        """Get the platform validator instance."""
        return self._platform_validator
    
    @property
    def generator_factory(self) -> GeneratorFactory:
        """Get the generator factory instance."""
        return self._generator_factory

    def check_cost_limit(
        self,
        estimate: CostEstimate,
        max_cost: float,
    ) -> tuple[bool, Optional[str]]:
        """Check if estimated cost is within limits.
        
        Implements cost control with 50% warning threshold.
        
        Args:
            estimate: Cost estimate to check
            max_cost: Maximum allowed cost in USD
            
        Returns:
            Tuple of (within_limit, warning_message)
            - within_limit: True if cost is within max_cost
            - warning_message: Warning if cost exceeds 50% threshold
        """
        # Check if cost exceeds limit
        if estimate.estimated_cost_usd > max_cost:
            return False, None
        
        # Check if cost exceeds warning threshold (50%)
        warning_threshold = max_cost * COST_WARNING_THRESHOLD
        if estimate.estimated_cost_usd > warning_threshold:
            percentage = (estimate.estimated_cost_usd / max_cost) * 100
            warning = (
                f"Cost estimate (${estimate.estimated_cost_usd:.4f}) is "
                f"{percentage:.1f}% of your max cost limit (${max_cost:.4f})"
            )
            return True, warning
        
        return True, None
    
    def format_estimate(self, estimate: CostEstimate) -> str:
        """Format cost estimate for display.
        
        Args:
            estimate: Cost estimate to format
            
        Returns:
            Formatted string for CLI display
        """
        lines = [
            "Cost Estimate:",
            f"  Provider: {estimate.provider}",
            f"  Model: {estimate.model}",
            f"  Estimated tokens: {estimate.estimated_tokens:,}",
            f"  Estimated cost: ${estimate.estimated_cost_usd:.4f}",
        ]
        
        if not estimate.within_budget:
            lines.append("  ⚠️  EXCEEDS BUDGET LIMIT")
        
        return "\n".join(lines)
    
    def dry_run(self, request: FormatRequest) -> Dict[str, Any]:
        """Perform a dry run to estimate cost without execution.
        
        This method estimates the cost of a formatting operation without
        actually making any LLM API calls.
        
        Args:
            request: FormatRequest to estimate
            
        Returns:
            Dictionary with cost estimate details
        """
        estimate = self.estimate_cost(request)
        
        result = {
            "dry_run": True,
            "output_type": request.output_type,
            "platform": request.platform,
            "llm_enhance": request.llm_enhance,
            "provider": estimate.provider,
            "model": estimate.model,
            "estimated_tokens": estimate.estimated_tokens,
            "estimated_cost_usd": estimate.estimated_cost_usd,
            "within_budget": estimate.within_budget,
        }
        
        if request.max_cost is not None:
            result["max_cost"] = request.max_cost
            within_limit, warning = self.check_cost_limit(estimate, request.max_cost)
            result["within_limit"] = within_limit
            if warning:
                result["warning"] = warning
        
        return result

    def format_bundle(
        self,
        bundle_name: str,
        enriched_content: Dict[str, Any],
        output_dir: Optional[str] = None,
        llm_enhance: bool = False,
        style_profile: Optional[str] = None,
        tone: Optional[str] = None,
        length: Optional[str] = None,
        provider: str = "bedrock",
        model: Optional[str] = None,
        max_cost: Optional[float] = None,
        dry_run: bool = False,
    ) -> BundleResult:
        """Generate all output types defined in a named bundle.
        
        Loads the bundle configuration by name and generates all output types
        defined in the bundle. Implements error isolation - if one output type
        fails, continues with the rest.
        
        Args:
            bundle_name: Name of the bundle to generate (e.g., "blog-launch")
            enriched_content: EnrichmentV1 content dictionary
            output_dir: Directory to write outputs (default: current directory)
            llm_enhance: Whether to use LLM enhancement
            style_profile: Style profile name or path
            tone: Tone override (professional, casual, technical, friendly)
            length: Length override (short, medium, long)
            provider: LLM provider (bedrock, openai, anthropic)
            model: Specific model to use
            max_cost: Maximum total cost in USD
            dry_run: If True, estimate cost without execution
            
        Returns:
            BundleResult with successful/failed outputs and manifest
            
        Raises:
            BundleNotFoundError: If bundle name doesn't exist
        """
        start_time = time.time()
        
        # Load bundle configuration (raises BundleNotFoundError if not found)
        bundle = self._bundle_loader.load_bundle(bundle_name)
        
        # Initialize result
        result = BundleResult(
            bundle_name=bundle_name,
            output_dir=output_dir or ".",
        )
        
        # Dry run - estimate total cost for all outputs
        if dry_run:
            total_estimated_cost = 0.0
            for output_type in bundle.outputs:
                request = FormatRequest(
                    enriched_content=enriched_content,
                    output_type=output_type,
                    llm_enhance=llm_enhance,
                    provider=provider,
                    model=model,
                )
                estimate = self.estimate_cost(request)
                total_estimated_cost += estimate.estimated_cost_usd
            
            result.total_cost = total_estimated_cost
            result.total_time = time.time() - start_time
            return result
        
        # Check total cost estimate against max_cost before starting
        if max_cost is not None and llm_enhance:
            total_estimated_cost = 0.0
            for output_type in bundle.outputs:
                request = FormatRequest(
                    enriched_content=enriched_content,
                    output_type=output_type,
                    llm_enhance=llm_enhance,
                    provider=provider,
                    model=model,
                )
                estimate = self.estimate_cost(request)
                total_estimated_cost += estimate.estimated_cost_usd
            
            if total_estimated_cost > max_cost:
                raise CostLimitExceededError(
                    message=f"Estimated bundle cost ${total_estimated_cost:.4f} exceeds limit ${max_cost:.4f}",
                    estimated_cost=total_estimated_cost,
                    max_cost=max_cost,
                    output_type=f"bundle:{bundle_name}",
                )
        
        # Create output directory if needed
        output_path = Path(output_dir) if output_dir else Path(".")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate each output type in the bundle
        total_cost = 0.0
        for output_type in bundle.outputs:
            try:
                # For promotional content types in bundles, use {{URL}} placeholder
                # since the actual URL won't be known until after publishing
                url_for_request = None
                if output_type in ["tweet", "linkedin", "newsletter"]:
                    url_for_request = "{{URL}}"
                
                request = FormatRequest(
                    enriched_content=enriched_content,
                    output_type=output_type,
                    llm_enhance=llm_enhance,
                    style_profile=style_profile,
                    tone=tone,
                    length=length,
                    provider=provider,
                    model=model,
                    url=url_for_request,
                )
                
                format_result = self.format_single(request)
                
                if format_result.success:
                    result.successful.append(output_type)
                    result.results[output_type] = format_result
                    
                    # Track cost if LLM was used
                    if format_result.metadata and format_result.metadata.llm_metadata:
                        total_cost += format_result.metadata.llm_metadata.cost_usd
                else:
                    result.failed.append((output_type, "Format generation failed"))
                    
            except Exception as e:
                # Error isolation - continue with other output types
                logger.warning(f"Failed to generate {output_type}: {e}")
                result.failed.append((output_type, str(e)))
        
        # Generate manifest file
        manifest = self._generate_manifest(
            bundle_name=bundle_name,
            bundle=bundle,
            result=result,
            enriched_content=enriched_content,
        )
        
        manifest_path = output_path / f"{bundle_name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        result.manifest_path = str(manifest_path)
        result.total_cost = total_cost
        result.total_time = time.time() - start_time
        
        return result
    
    def _generate_manifest(
        self,
        bundle_name: str,
        bundle: BundleConfig,
        result: BundleResult,
        enriched_content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate manifest file for bundle output.
        
        Args:
            bundle_name: Name of the bundle
            bundle: Bundle configuration
            result: Bundle generation result
            enriched_content: Source enriched content
            
        Returns:
            Manifest dictionary
        """
        return {
            "manifest_version": "v1",
            "bundle_name": bundle_name,
            "bundle_description": bundle.description,
            "timestamp": datetime.now().isoformat(),
            "source_file": enriched_content.get("source_file", ""),
            "outputs_requested": bundle.outputs,
            "outputs_successful": result.successful,
            "outputs_failed": [
                {"output_type": ot, "error": err}
                for ot, err in result.failed
            ],
            "total_cost_usd": result.total_cost,
            "total_time_seconds": result.total_time,
            "files_generated": {
                output_type: f"{output_type}.md"
                for output_type in result.successful
            },
        }
    
    def list_bundles(self) -> List[BundleConfig]:
        """List all available bundles.
        
        Returns:
            List of BundleConfig objects
        """
        return self._bundle_loader.list_bundles()
    
    def get_bundle_names(self) -> List[str]:
        """Get list of available bundle names.
        
        Returns:
            Sorted list of bundle names
        """
        return self._bundle_loader.get_bundle_names()
    
    def has_bundle(self, name: str) -> bool:
        """Check if a bundle exists.
        
        Args:
            name: Bundle name to check
            
        Returns:
            True if bundle exists
        """
        return self._bundle_loader.has_bundle(name)
    
    def format_bundle_list(self) -> str:
        """Format bundle list for CLI display.
        
        Returns:
            Formatted string showing all bundles
        """
        return self._bundle_loader.format_bundle_list()
    
    @property
    def bundle_loader(self) -> BundleLoader:
        """Get the bundle loader instance."""
        return self._bundle_loader
    
    @bundle_loader.setter
    def bundle_loader(self, loader: BundleLoader) -> None:
        """Set the bundle loader instance."""
        self._bundle_loader = loader

    def format_batch(
        self,
        input_pattern: str,
        output_type: str,
        output_dir: Optional[str] = None,
        llm_enhance: bool = False,
        style_profile: Optional[str] = None,
        tone: Optional[str] = None,
        length: Optional[str] = None,
        platform: Optional[str] = None,
        provider: str = "bedrock",
        model: Optional[str] = None,
        max_cost: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> BatchResult:
        """Process multiple input files matching a glob pattern.
        
        Implements batch processing with error isolation - if one file fails,
        continues with the rest.
        
        Args:
            input_pattern: Glob pattern for input files (e.g., "*.enriched.json")
            output_type: Output type to generate for each file
            output_dir: Directory to write outputs (default: same as input)
            llm_enhance: Whether to use LLM enhancement
            style_profile: Style profile name or path
            tone: Tone override
            length: Length override
            platform: Target platform for validation
            provider: LLM provider
            model: Specific model to use
            max_cost: Maximum total cost in USD for entire batch
            progress_callback: Optional callback(current, total, file_path) for progress
            
        Returns:
            BatchResult with successful/failed files and summary
        """
        import glob
        
        start_time = time.time()
        
        # Find all matching files
        input_files = sorted(glob.glob(input_pattern, recursive=True))
        
        if not input_files:
            return BatchResult(
                output_dir=output_dir or ".",
                total_time=time.time() - start_time,
            )
        
        # Initialize result
        result = BatchResult(
            output_dir=output_dir or ".",
        )
        
        # Estimate total cost if max_cost is set
        if max_cost is not None and llm_enhance:
            total_estimated_cost = 0.0
            for file_path in input_files:
                try:
                    enriched_content = self._load_enriched_file(file_path)
                    request = FormatRequest(
                        enriched_content=enriched_content,
                        output_type=output_type,
                        llm_enhance=llm_enhance,
                        provider=provider,
                        model=model,
                    )
                    estimate = self.estimate_cost(request)
                    total_estimated_cost += estimate.estimated_cost_usd
                except Exception:
                    # Skip files that can't be loaded for estimation
                    pass
            
            if total_estimated_cost > max_cost:
                raise CostLimitExceededError(
                    message=f"Estimated batch cost ${total_estimated_cost:.4f} exceeds limit ${max_cost:.4f}",
                    estimated_cost=total_estimated_cost,
                    max_cost=max_cost,
                    output_type=f"batch:{output_type}",
                )
        
        # Create output directory if needed
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        total_cost = 0.0
        total_files = len(input_files)
        
        for idx, file_path in enumerate(input_files):
            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, total_files, file_path)
            
            try:
                # Load enriched content
                enriched_content = self._load_enriched_file(file_path)
                
                # Create format request
                request = FormatRequest(
                    enriched_content=enriched_content,
                    output_type=output_type,
                    platform=platform,
                    llm_enhance=llm_enhance,
                    style_profile=style_profile,
                    tone=tone,
                    length=length,
                    provider=provider,
                    model=model,
                )
                
                # Format the content
                format_result = self.format_single(request)
                
                if format_result.success:
                    result.successful.append(file_path)
                    result.results[file_path] = format_result
                    
                    # Track cost if LLM was used
                    if format_result.metadata and format_result.metadata.llm_metadata:
                        total_cost += format_result.metadata.llm_metadata.cost_usd
                else:
                    result.failed.append((file_path, "Format generation failed"))
                    
            except Exception as e:
                # Error isolation - continue with other files
                logger.warning(f"Failed to process {file_path}: {e}")
                result.failed.append((file_path, str(e)))
        
        result.total_cost = total_cost
        result.total_time = time.time() - start_time
        
        return result
    
    def _load_enriched_file(self, file_path: str) -> Dict[str, Any]:
        """Load enriched content from a JSON file.
        
        Args:
            file_path: Path to the enriched JSON file
            
        Returns:
            Enriched content dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def format_batch_summary(self, result: BatchResult) -> str:
        """Format batch result summary for CLI display.
        
        Args:
            result: BatchResult to format
            
        Returns:
            Formatted summary string
        """
        lines = [
            "Batch Processing Summary:",
            f"  Total files: {len(result.successful) + len(result.failed)}",
            f"  Successful: {len(result.successful)}",
            f"  Failed: {len(result.failed)}",
            f"  Total cost: ${result.total_cost:.4f}",
            f"  Total time: {result.total_time:.2f}s",
        ]
        
        if result.failed:
            lines.append("")
            lines.append("  Failed files:")
            for file_path, error in result.failed:
                lines.append(f"    - {file_path}: {error}")
        
        return "\n".join(lines)
