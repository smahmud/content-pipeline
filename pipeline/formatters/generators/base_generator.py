"""
Base generator class for output type formatters.

Provides common functionality for all generators including:
- Template rendering via TemplateEngine
- Input validation
- Metadata generation
- Platform validation integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pipeline.formatters.base import (
    BaseFormatter,
    FormatRequest,
    FormatResult,
    OutputType,
)
from pipeline.formatters.schemas.format_v1 import FormatV1, ValidationMetadata
from pipeline.formatters.template_engine import TemplateEngine
from pipeline.formatters.validator import PlatformValidator


@dataclass
class GeneratorConfig:
    """Configuration for a generator.
    
    Attributes:
        template_engine: TemplateEngine instance for rendering
        platform_validator: PlatformValidator for content validation
        auto_truncate: Whether to auto-truncate content exceeding limits
    """
    
    template_engine: TemplateEngine
    platform_validator: PlatformValidator
    auto_truncate: bool = True


class BaseGenerator(ABC):
    """Abstract base class for all output type generators.
    
    Provides common functionality and enforces the BaseFormatter protocol.
    Subclasses must implement:
    - output_type property
    - supported_platforms property
    - required_enrichments property
    - _build_template_context() method
    """
    
    def __init__(self, config: GeneratorConfig):
        """Initialize the generator.
        
        Args:
            config: GeneratorConfig with template engine and validator
        """
        self.config = config
        self._template_engine = config.template_engine
        self._validator = config.platform_validator
    
    @property
    @abstractmethod
    def output_type(self) -> str:
        """Return the output type this generator handles."""
        ...
    
    @property
    @abstractmethod
    def supported_platforms(self) -> list[str]:
        """Return list of supported platforms for this output type."""
        ...
    
    @property
    @abstractmethod
    def required_enrichments(self) -> list[str]:
        """Return list of required enrichment fields."""
        ...
    
    @abstractmethod
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build the context dictionary for template rendering.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Dictionary to pass to template engine
        """
        ...
    
    def format(self, request: FormatRequest) -> FormatResult:
        """Generate formatted output from enriched content.
        
        Args:
            request: FormatRequest with enriched content and options
            
        Returns:
            FormatResult with formatted content and metadata
        """
        # Validate input
        is_valid, errors = self.validate_input(request.enriched_content)
        if not is_valid:
            return FormatResult(
                content="",
                metadata=self._create_error_metadata(request, errors),
                warnings=errors,
                success=False,
                error=f"Input validation failed: {'; '.join(errors)}",
            )
        
        # Build template context
        context = self._build_template_context(
            request.enriched_content,
            request,
        )
        
        # Render template
        try:
            content = self._template_engine.render(self.output_type, context)
        except Exception as e:
            return FormatResult(
                content="",
                metadata=self._create_error_metadata(request, [str(e)]),
                warnings=[str(e)],
                success=False,
                error=f"Template rendering failed: {e}",
            )
        
        # Validate and optionally truncate for platform
        warnings: list[str] = []
        truncated = False
        
        if request.platform:
            content, validation_result = self._validator.validate_and_truncate(
                content,
                request.platform,
                auto_truncate=self.config.auto_truncate,
            )
            warnings.extend(validation_result.warnings)
            truncated = validation_result.truncated
        
        # Create metadata
        metadata = self._create_metadata(
            request=request,
            content=content,
            truncated=truncated,
            warnings=warnings,
        )
        
        return FormatResult(
            content=content,
            metadata=metadata,
            warnings=warnings,
            success=True,
        )
    
    def validate_input(self, enriched_content: dict) -> tuple[bool, list[str]]:
        """Validate that enriched content has required fields.
        
        Args:
            enriched_content: EnrichmentV1 data dictionary
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        
        # Check for required enrichment fields
        for field_name in self.required_enrichments:
            if not self._has_field(enriched_content, field_name):
                errors.append(f"Missing required enrichment: {field_name}")
        
        return len(errors) == 0, errors
    
    def _has_field(self, data: dict, field_path: str) -> bool:
        """Check if a nested field exists in the data.
        
        Args:
            data: Dictionary to check
            field_path: Dot-separated path (e.g., "summary.short")
            
        Returns:
            True if field exists and is not None/empty
        """
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if not isinstance(current, dict):
                return False
            if part not in current:
                return False
            current = current[part]
        
        # Check if value is not None/empty
        if current is None:
            return False
        if isinstance(current, str) and not current.strip():
            return False
        if isinstance(current, (list, dict)) and len(current) == 0:
            return False
        
        return True
    
    def _get_field(
        self,
        data: dict,
        field_path: str,
        default: Any = None,
    ) -> Any:
        """Get a nested field value from the data.
        
        Args:
            data: Dictionary to get from
            field_path: Dot-separated path (e.g., "summary.short")
            default: Default value if field not found
            
        Returns:
            Field value or default
        """
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if not isinstance(current, dict):
                return default
            if part not in current:
                return default
            current = current[part]
        
        return current if current is not None else default
    
    def _create_metadata(
        self,
        request: FormatRequest,
        content: str,
        truncated: bool = False,
        warnings: Optional[list[str]] = None,
    ) -> FormatV1:
        """Create FormatV1 metadata for the output.
        
        Args:
            request: The format request
            content: The generated content
            truncated: Whether content was truncated
            warnings: List of warnings
            
        Returns:
            FormatV1 metadata object
        """
        # Get source file from enriched content metadata
        source_file = self._get_field(
            request.enriched_content,
            "metadata.source_file",
            "unknown",
        )
        
        return FormatV1(
            output_type=self.output_type,
            platform=request.platform,
            timestamp=datetime.now(timezone.utc),
            source_file=source_file,
            style_profile_used=(
                request.style_profile.get("name")
                if request.style_profile
                else None
            ),
            llm_metadata=None,  # Set by LLM enhancer if used
            validation=ValidationMetadata(
                platform=request.platform,
                character_count=len(content),
                truncated=truncated,
                warnings=warnings or [],
            ),
            tone=request.tone,
            length=request.length,
        )
    
    def _create_error_metadata(
        self,
        request: FormatRequest,
        errors: list[str],
    ) -> FormatV1:
        """Create FormatV1 metadata for an error case.
        
        Args:
            request: The format request
            errors: List of error messages
            
        Returns:
            FormatV1 metadata object
        """
        source_file = self._get_field(
            request.enriched_content,
            "metadata.source_file",
            "unknown",
        )
        
        return FormatV1(
            output_type=self.output_type,
            platform=request.platform,
            timestamp=datetime.now(timezone.utc),
            source_file=source_file,
            validation=ValidationMetadata(
                platform=request.platform,
                character_count=0,
                truncated=False,
                warnings=errors,
            ),
        )
