"""
Formatter module for generating platform-specific publishing formats.

This module transforms enriched JSON content into 16 platform-specific
publishing formats using a hybrid architecture that combines Jinja2 
templates for structural consistency with LLM-powered prose enhancement.
"""

from pipeline.formatters.base import (
    BaseFormatter,
    FormatRequest,
    FormatResult,
    OutputType,
)
from pipeline.formatters.style_profile import (
    StyleProfile,
    StyleProfileError,
    StyleProfileLoader,
)
from pipeline.formatters.template_engine import (
    TemplateEngine,
    TemplateEngineError,
)
from pipeline.formatters.validator import (
    PlatformLimits,
    PlatformValidator,
    PlatformValidatorError,
    ValidationResult,
)
from pipeline.formatters.generators import (
    BaseGenerator,
    GeneratorConfig,
    GeneratorFactory,
    GeneratorFactoryError,
)
from pipeline.formatters.llm import (
    LLMEnhancer,
    EnhancementConfig,
    EnhancementResult,
)
from pipeline.formatters.llm.prompts import PromptLoader
from pipeline.formatters.errors import (
    FormatError,
    InputValidationError,
    TemplateError,
    EnhancementError,
    PlatformValidationError,
    GeneratorError,
    CostLimitExceededError,
    OutputWriteError,
    BatchProcessingError,
    BundleGenerationError,
    FormatErrorInfo,
)
from pipeline.formatters.retry import (
    retry_enhancement,
    retry_with_fallback,
    EnhancementRetryContext,
    is_transient_error,
    is_permanent_error,
    TRANSIENT_ERRORS,
    PERMANENT_ERRORS,
)
from pipeline.formatters.input_validator import (
    InputValidator,
    ValidationResult as InputValidationResult,
    validate_input,
    validate_input_file,
    get_required_enrichments_for_type,
    REQUIRED_ENRICHMENTS,
)

__all__ = [
    # Base classes
    "BaseFormatter",
    "FormatRequest",
    "FormatResult",
    "OutputType",
    # Generator infrastructure
    "BaseGenerator",
    "GeneratorConfig",
    "GeneratorFactory",
    "GeneratorFactoryError",
    # Platform validation
    "PlatformLimits",
    "PlatformValidator",
    "PlatformValidatorError",
    "ValidationResult",
    # Style profiles
    "StyleProfile",
    "StyleProfileError",
    "StyleProfileLoader",
    # Template engine
    "TemplateEngine",
    "TemplateEngineError",
    # LLM enhancement
    "LLMEnhancer",
    "EnhancementConfig",
    "EnhancementResult",
    "PromptLoader",
    # Error handling
    "FormatError",
    "InputValidationError",
    "TemplateError",
    "EnhancementError",
    "PlatformValidationError",
    "GeneratorError",
    "CostLimitExceededError",
    "OutputWriteError",
    "BatchProcessingError",
    "BundleGenerationError",
    "FormatErrorInfo",
    # Retry utilities
    "retry_enhancement",
    "retry_with_fallback",
    "EnhancementRetryContext",
    "is_transient_error",
    "is_permanent_error",
    "TRANSIENT_ERRORS",
    "PERMANENT_ERRORS",
    # Input validation
    "InputValidator",
    "InputValidationResult",
    "validate_input",
    "validate_input_file",
    "get_required_enrichments_for_type",
    "REQUIRED_ENRICHMENTS",
]
