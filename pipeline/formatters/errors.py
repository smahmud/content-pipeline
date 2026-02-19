"""
Formatter Error Hierarchy

Defines all custom exceptions used in the formatter system.
This provides clear, specific error types for different failure scenarios.

Error Categories:
- Input Errors: Missing file, invalid JSON, schema validation failure
- Configuration Errors: Missing template, invalid style profile, missing API key
- Transient Errors: Rate limits, timeouts, network errors (retryable)
- Permanent Errors: Authentication failure, invalid request (not retryable)
- Enhancement Errors: LLM failures after retries (triggers graceful degradation)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


class FormatError(Exception):
    """Base exception for all formatter errors.
    
    All formatter-specific exceptions inherit from this class,
    allowing for broad exception handling when needed.
    """
    pass


class InputValidationError(FormatError):
    """Error validating input data.
    
    Raised when:
    - Input file is missing or cannot be read
    - Input JSON is malformed
    - Input does not conform to EnrichmentV1 schema
    - Required enrichment fields are missing for the requested output type
    
    Attributes:
        field_name: Name of the field that failed validation (if applicable)
        expected: Expected value or type
        actual: Actual value or type received
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
    ):
        super().__init__(message)
        self.field_name = field_name
        self.expected = expected
        self.actual = actual


class TemplateError(FormatError):
    """Error loading or rendering templates.
    
    Raised when:
    - Template file is missing
    - Template has syntax errors
    - Template rendering fails due to missing variables
    - Custom template is invalid
    
    Attributes:
        template_name: Name of the template that failed
        template_path: Path to the template file (if applicable)
        original_error: The underlying Jinja2 error
    """
    
    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        template_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.template_name = template_name
        self.template_path = template_path
        self.original_error = original_error


class StyleProfileError(FormatError):
    """Error loading or validating style profiles.
    
    Raised when:
    - Style profile file is missing
    - Style profile has invalid YAML frontmatter
    - Style profile is missing required fields
    - Style profile has invalid field values
    
    Attributes:
        profile_name: Name of the style profile
        profile_path: Path to the profile file
        missing_fields: List of missing required fields
    """
    
    def __init__(
        self,
        message: str,
        profile_name: Optional[str] = None,
        profile_path: Optional[str] = None,
        missing_fields: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.profile_name = profile_name
        self.profile_path = profile_path
        self.missing_fields = missing_fields or []


class EnhancementError(FormatError):
    """Error during LLM enhancement.
    
    Raised when LLM enhancement fails after all retries are exhausted.
    This error triggers graceful degradation to template-only output.
    
    Attributes:
        provider: LLM provider that was used
        model: Model that was used
        original_error: The underlying error from the LLM provider
        attempts: Number of retry attempts made
        recoverable: Whether the error can be recovered from (via fallback)
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
        attempts: int = 0,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_error = original_error
        self.attempts = attempts
        self.recoverable = recoverable


class PlatformValidationError(FormatError):
    """Error validating content against platform constraints.
    
    Raised when content cannot be validated or truncated to fit
    platform requirements.
    
    Attributes:
        platform: Platform that validation failed for
        constraint: The constraint that was violated
        actual_value: The actual value that violated the constraint
        limit: The limit that was exceeded
    """
    
    def __init__(
        self,
        message: str,
        platform: Optional[str] = None,
        constraint: Optional[str] = None,
        actual_value: Optional[Any] = None,
        limit: Optional[Any] = None,
    ):
        super().__init__(message)
        self.platform = platform
        self.constraint = constraint
        self.actual_value = actual_value
        self.limit = limit


class GeneratorError(FormatError):
    """Error in output type generator.
    
    Raised when a generator fails to produce output.
    
    Attributes:
        output_type: The output type that failed
        generator_name: Name of the generator class
        original_error: The underlying error
    """
    
    def __init__(
        self,
        message: str,
        output_type: Optional[str] = None,
        generator_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.output_type = output_type
        self.generator_name = generator_name
        self.original_error = original_error


class CostLimitExceededError(FormatError):
    """Estimated cost exceeds user-specified limit.
    
    Raised when the estimated cost of a formatting operation exceeds
    the --max-cost limit specified by the user.
    
    Attributes:
        estimated_cost: The estimated cost in USD
        max_cost: The maximum cost limit in USD
        output_type: The output type being generated
    """
    
    def __init__(
        self,
        message: str,
        estimated_cost: float = 0.0,
        max_cost: float = 0.0,
        output_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.estimated_cost = estimated_cost
        self.max_cost = max_cost
        self.output_type = output_type


class OutputWriteError(FormatError):
    """Error writing output file.
    
    Raised when output file cannot be written due to permissions,
    disk space, or other file system issues.
    
    Attributes:
        output_path: Path where write was attempted
        original_error: The underlying file system error
    """
    
    def __init__(
        self,
        message: str,
        output_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.output_path = output_path
        self.original_error = original_error


class BatchProcessingError(FormatError):
    """Error during batch processing.
    
    Raised when batch processing encounters an error that affects
    the entire batch operation (not individual file failures).
    
    Attributes:
        pattern: The glob pattern used for batch processing
        files_processed: Number of files processed before error
        total_files: Total number of files to process
    """
    
    def __init__(
        self,
        message: str,
        pattern: Optional[str] = None,
        files_processed: int = 0,
        total_files: int = 0,
    ):
        super().__init__(message)
        self.pattern = pattern
        self.files_processed = files_processed
        self.total_files = total_files


class BundleGenerationError(FormatError):
    """Error during bundle generation.
    
    Raised when bundle generation fails completely (not individual
    output type failures, which are handled with error isolation).
    
    Attributes:
        output_dir: Directory where bundle was being generated
        types_completed: List of output types completed before error
        types_remaining: List of output types not yet attempted
    """
    
    def __init__(
        self,
        message: str,
        output_dir: Optional[str] = None,
        types_completed: Optional[List[str]] = None,
        types_remaining: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.output_dir = output_dir
        self.types_completed = types_completed or []
        self.types_remaining = types_remaining or []


@dataclass
class FormatErrorInfo:
    """Structured error information for user-friendly error reporting.
    
    This dataclass provides a standardized format for error information
    that can be displayed to users or logged for debugging.
    
    Attributes:
        error_type: Type of error (e.g., "InputValidationError")
        message: Human-readable error message
        details: Additional context (file path, field name, etc.)
        recoverable: Whether the error can be recovered from
        suggestion: Suggested action for the user
    """
    error_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    suggestion: str = ""
    
    @classmethod
    def from_exception(cls, error: Exception) -> "FormatErrorInfo":
        """Create FormatErrorInfo from an exception.
        
        Args:
            error: Exception to convert
            
        Returns:
            FormatErrorInfo with details extracted from the exception
        """
        error_type = type(error).__name__
        message = str(error)
        details = {}
        recoverable = False
        suggestion = ""
        
        if isinstance(error, InputValidationError):
            if error.field_name:
                details["field_name"] = error.field_name
            if error.expected:
                details["expected"] = error.expected
            if error.actual:
                details["actual"] = error.actual
            suggestion = "Check that the input file is valid JSON conforming to EnrichmentV1 schema."
        
        elif isinstance(error, TemplateError):
            if error.template_name:
                details["template_name"] = error.template_name
            if error.template_path:
                details["template_path"] = error.template_path
            suggestion = "Verify the template exists and has valid Jinja2 syntax."
        
        elif isinstance(error, StyleProfileError):
            if error.profile_name:
                details["profile_name"] = error.profile_name
            if error.profile_path:
                details["profile_path"] = error.profile_path
            if error.missing_fields:
                details["missing_fields"] = error.missing_fields
            suggestion = "Check that the style profile has valid YAML frontmatter with all required fields."
        
        elif isinstance(error, EnhancementError):
            if error.provider:
                details["provider"] = error.provider
            if error.model:
                details["model"] = error.model
            details["attempts"] = error.attempts
            recoverable = error.recoverable
            if recoverable:
                suggestion = "LLM enhancement failed. Template-only output will be used as fallback."
            else:
                suggestion = "Check your API credentials and network connection."
        
        elif isinstance(error, PlatformValidationError):
            if error.platform:
                details["platform"] = error.platform
            if error.constraint:
                details["constraint"] = error.constraint
            if error.actual_value is not None:
                details["actual_value"] = error.actual_value
            if error.limit is not None:
                details["limit"] = error.limit
            suggestion = "Content will be truncated to fit platform limits."
            recoverable = True
        
        elif isinstance(error, CostLimitExceededError):
            details["estimated_cost"] = error.estimated_cost
            details["max_cost"] = error.max_cost
            if error.output_type:
                details["output_type"] = error.output_type
            suggestion = "Increase --max-cost limit or use --no-llm to skip LLM enhancement."
        
        elif isinstance(error, OutputWriteError):
            if error.output_path:
                details["output_path"] = error.output_path
            suggestion = "Check file permissions and available disk space."
        
        elif isinstance(error, BatchProcessingError):
            if error.pattern:
                details["pattern"] = error.pattern
            details["files_processed"] = error.files_processed
            details["total_files"] = error.total_files
            suggestion = "Check the batch pattern and ensure input files are accessible."
        
        elif isinstance(error, BundleGenerationError):
            if error.output_dir:
                details["output_dir"] = error.output_dir
            details["types_completed"] = error.types_completed
            details["types_remaining"] = error.types_remaining
            suggestion = "Check the output directory permissions and available disk space."
        
        return cls(
            error_type=error_type,
            message=message,
            details=details,
            recoverable=recoverable,
            suggestion=suggestion,
        )
