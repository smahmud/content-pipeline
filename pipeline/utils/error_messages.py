"""
Centralized Error Message Templates

This module provides consistent error message formatting and templates
for all error categories in the content pipeline system.

Enhanced in v0.6.5 to support:
- Consistent error message formatting across all components
- Actionable suggestions and examples in error messages
- Specific error messages for each failure category
- Integration with logging and debugging support
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging


class ErrorCategory(Enum):
    """Categories of errors that can occur in the system."""
    CONFIGURATION = "configuration"
    ENGINE_INITIALIZATION = "engine_initialization"
    API_AUTHENTICATION = "api_authentication"
    FILE_ACCESS = "file_access"
    NETWORK = "network"
    VALIDATION = "validation"
    BREAKING_CHANGE = "breaking_change"
    GENERAL = "general"


class ErrorMessages:
    """
    Centralized error message templates with consistent formatting.
    
    Provides specific error messages for each failure category with
    actionable suggestions and examples.
    """
    
    # Configuration Error Templates
    CONFIGURATION_TEMPLATES = {
        "invalid_yaml": """
Configuration Error: Invalid YAML syntax in {file_path}

Line {line_number}: {error_details}

Suggestions:
  • Check for proper indentation (use spaces, not tabs)
  • Ensure all quotes are properly closed
  • Validate YAML syntax online: https://yamlchecker.com/
  
Example valid configuration:
  engine: local-whisper
  output_dir: ./transcripts
  whisper_local:
    model: base
""",
        
        "missing_required_field": """
Configuration Error: Missing required field '{field_name}'

The configuration file is missing a required field.

Suggestions:
  • Add the missing field to your configuration file
  • Use --config to specify a different configuration file
  • Run with --help to see all available options
  
Example:
  {field_name}: {example_value}
""",
        
        "invalid_field_value": """
Configuration Error: Invalid value for '{field_name}'

Current value: {current_value}
Expected: {expected_values}

Suggestions:
  • Update the configuration file with a valid value
  • Check the documentation for valid options
  • Use CLI flags to override configuration values
  
Example:
  {field_name}: {example_value}
""",
        
        "environment_variable_missing": """
Configuration Error: Environment variable '{variable_name}' not found

The configuration references ${{{variable_name}}} but this environment variable is not set.

Suggestions:
  • Set the environment variable: export {variable_name}=your_value
  • Use a default value: ${{{variable_name}:-default_value}}
  • Provide the value directly in the configuration file
  
Example:
  export {variable_name}="your_actual_value"
"""
    }
    
    # Engine Initialization Error Templates
    ENGINE_TEMPLATES = {
        "engine_not_available": """
Engine Error: '{engine_name}' engine is not available

{specific_reason}

Suggestions:
  • Install required dependencies: {installation_command}
  • Try a different engine: --engine auto
  • Check system requirements and compatibility
  
Available engines: {available_engines}
""",
        
        "model_not_found": """
Engine Error: Model '{model_name}' not found for {engine_name}

The specified model is not available or not downloaded.

Suggestions:
  • Use a different model: {available_models}
  • Download the model (it will be downloaded automatically on first use)
  • Check your internet connection for model downloads
  
Example:
  --model base  # Use the base model instead
""",
        
        "engine_initialization_failed": """
Engine Error: Failed to initialize {engine_name} engine

{error_details}

Suggestions:
  • Check system requirements and dependencies
  • Verify configuration settings for this engine
  • Try restarting with --log-level debug for more details
  • Consider using --engine auto to select an alternative
  
Troubleshooting:
  1. Verify all dependencies are installed
  2. Check available system resources (memory, disk space)
  3. Review engine-specific configuration settings
"""
    }
    
    # API Authentication Error Templates
    API_TEMPLATES = {
        "missing_api_key": """
Authentication Error: API key required for {service_name}

No API key found for {service_name} service.

Setup Instructions:
  Option 1: Environment variable
    export {env_var_name}="your-api-key-here"
  
  Option 2: CLI flag
    --api-key your-api-key-here
  
  Option 3: Configuration file
    {config_section}:
      api_key: ${{{env_var_name}}}

Get your API key: {api_key_url}
""",
        
        "invalid_api_key": """
Authentication Error: Invalid API key for {service_name}

The provided API key is not valid or has expired.

Suggestions:
  • Verify your API key is correct and active
  • Check if your API key has the required permissions
  • Generate a new API key if needed
  • Ensure no extra spaces or characters in the key
  
Get a new API key: {api_key_url}
""",
        
        "api_quota_exceeded": """
Authentication Error: API quota exceeded for {service_name}

You have exceeded your API usage limits.

Suggestions:
  • Check your usage dashboard: {dashboard_url}
  • Upgrade your plan if needed
  • Wait for quota reset (usually monthly)
  • Consider using local alternatives: --engine local-whisper
  
Alternative:
  # Use local processing instead
  --engine local-whisper --model base
"""
    }
    
    # File Access Error Templates
    FILE_TEMPLATES = {
        "file_not_found": """
File Error: File not found '{file_path}'

The specified file does not exist or cannot be accessed.

Suggestions:
  • Check the file path is correct
  • Verify the file exists: ls -la "{file_path}"
  • Use absolute path if relative path is not working
  • Check file permissions and ownership
  
Example:
  # Use absolute path
  --source /full/path/to/your/audio.mp3
""",
        
        "permission_denied": """
File Error: Permission denied accessing '{file_path}'

You don't have the required permissions to access this file.

Suggestions:
  • Check file permissions: ls -la "{file_path}"
  • Change permissions: chmod 644 "{file_path}"
  • Run as appropriate user or use sudo if necessary
  • Ensure the file is not locked by another process
  
Fix permissions:
  chmod 644 "{file_path}"  # For read access
  chmod 755 "{file_path}"  # For execute access
""",
        
        "directory_creation_failed": """
File Error: Cannot create directory '{directory_path}'

Failed to create the output directory.

Suggestions:
  • Check parent directory permissions
  • Ensure sufficient disk space
  • Verify the path is valid for your operating system
  • Try creating the directory manually first
  
Manual creation:
  mkdir -p "{directory_path}"
""",
        
        "file_write_failed": """
File Error: Cannot write to file '{file_path}'

Failed to write the output file.

Suggestions:
  • Check directory permissions for the output location
  • Ensure sufficient disk space
  • Verify the file is not locked or in use
  • Try a different output location
  
Alternative:
  --output-dir /tmp/transcripts  # Use temporary directory
"""
    }
    
    # Network Error Templates
    NETWORK_TEMPLATES = {
        "connection_timeout": """
Network Error: Connection timeout to {service_name}

The request timed out while connecting to the service.

Retry Strategies:
  • Check your internet connection
  • Try again in a few minutes (service may be busy)
  • Use a different network if available
  • Consider using offline alternatives
  
Offline Alternative:
  --engine local-whisper  # Process locally without internet
""",
        
        "connection_failed": """
Network Error: Failed to connect to {service_name}

Cannot establish connection to the remote service.

Suggestions:
  • Check your internet connection
  • Verify the service is not down: {status_url}
  • Check firewall and proxy settings
  • Try using a VPN if access is restricted
  
Troubleshooting:
  1. Test connection: ping {hostname}
  2. Check service status: {status_url}
  3. Verify DNS resolution
""",
        
        "rate_limit_exceeded": """
Network Error: Rate limit exceeded for {service_name}

You are making requests too quickly.

Suggestions:
  • Wait {retry_after} seconds before retrying
  • Reduce request frequency
  • Consider upgrading your API plan
  • Use local processing to avoid rate limits
  
Alternative:
  --engine local-whisper  # Avoid rate limits entirely
"""
    }
    
    # Validation Error Templates
    VALIDATION_TEMPLATES = {
        "invalid_input_format": """
Validation Error: Invalid input format for '{field_name}'

Current value: {current_value}
Expected format: {expected_format}

Suggestions:
  • Check the input format matches the expected pattern
  • Review the documentation for valid formats
  • Use the provided examples as a reference
  
Valid examples:
{examples}
""",
        
        "value_out_of_range": """
Validation Error: Value out of range for '{field_name}'

Current value: {current_value}
Valid range: {min_value} to {max_value}

Suggestions:
  • Use a value within the valid range
  • Check the documentation for acceptable values
  • Consider using the default value if unsure
  
Example:
  {field_name}: {suggested_value}
"""
    }
    
    @classmethod
    def format_error(
        cls,
        category: ErrorCategory,
        template_key: str,
        **kwargs
    ) -> str:
        """
        Format an error message using the specified template.
        
        Args:
            category: The error category
            template_key: The specific template within the category
            **kwargs: Template variables to substitute
            
        Returns:
            Formatted error message with suggestions
        """
        templates = cls._get_templates_for_category(category)
        
        if template_key not in templates:
            return cls._format_generic_error(category, template_key, **kwargs)
        
        template = templates[template_key]
        
        try:
            return template.format(**kwargs).strip()
        except KeyError as e:
            logging.warning(f"Missing template variable {e} for {category.value}.{template_key}")
            return cls._format_generic_error(category, template_key, **kwargs)
    
    @classmethod
    def _get_templates_for_category(cls, category: ErrorCategory) -> Dict[str, str]:
        """Get templates for a specific error category."""
        template_map = {
            ErrorCategory.CONFIGURATION: cls.CONFIGURATION_TEMPLATES,
            ErrorCategory.ENGINE_INITIALIZATION: cls.ENGINE_TEMPLATES,
            ErrorCategory.API_AUTHENTICATION: cls.API_TEMPLATES,
            ErrorCategory.FILE_ACCESS: cls.FILE_TEMPLATES,
            ErrorCategory.NETWORK: cls.NETWORK_TEMPLATES,
            ErrorCategory.VALIDATION: cls.VALIDATION_TEMPLATES,
        }
        
        return template_map.get(category, {})
    
    @classmethod
    def _format_generic_error(
        cls,
        category: ErrorCategory,
        template_key: str,
        **kwargs
    ) -> str:
        """Format a generic error message when specific template is not found."""
        error_details = kwargs.get('error_details', 'Unknown error occurred')
        
        return f"""
{category.value.replace('_', ' ').title()} Error: {template_key}

{error_details}

General Suggestions:
  • Check the logs for more detailed error information
  • Try running with --log-level debug for additional details
  • Verify your configuration and input parameters
  • Consult the documentation for troubleshooting guidance

For help: content-pipeline --help
"""
    
    @classmethod
    def get_suggestion_for_error(cls, error_message: str) -> Optional[str]:
        """
        Analyze an error message and provide contextual suggestions.
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Suggested action or None if no specific suggestion available
        """
        error_lower = error_message.lower()
        
        # Configuration errors
        if "yaml" in error_lower or "configuration" in error_lower:
            return "Check your configuration file syntax and required fields"
        
        # Engine errors
        elif "engine" in error_lower and "not available" in error_lower:
            return "Try using --engine auto to automatically select an available engine"
        
        # API errors
        elif "api" in error_lower or "authentication" in error_lower:
            return "Verify your API credentials and network connection"
        
        # File errors
        elif "file" in error_lower and ("not found" in error_lower or "permission" in error_lower):
            return "Check file paths and permissions"
        
        # Network errors
        elif "network" in error_lower or "connection" in error_lower:
            return "Check your internet connection and try again"
        
        else:
            return "Run with --log-level debug for more detailed error information"


class ErrorFormatter:
    """
    Utility class for formatting errors with consistent styling.
    """
    
    @staticmethod
    def format_for_cli(error_message: str, include_emoji: bool = False) -> str:
        """
        Format error message for CLI output with consistent styling.
        
        Args:
            error_message: The error message to format
            include_emoji: Whether to include emoji indicators (disabled for Windows compatibility)
            
        Returns:
            Formatted error message for CLI display
        """
        # Emoji support disabled for Windows compatibility
        return error_message
    
    @staticmethod
    def format_for_logging(
        error_message: str,
        level: str = "ERROR",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format error message for logging with additional context.
        
        Args:
            error_message: The error message to format
            level: The log level
            context: Additional context information
            
        Returns:
            Formatted error message for logging
        """
        formatted = f"[{level}] {error_message}"
        
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            formatted += f" | Context: {context_str}"
        
        return formatted
    
    @staticmethod
    def extract_actionable_suggestions(error_message: str) -> List[str]:
        """
        Extract actionable suggestions from an error message.
        
        Args:
            error_message: The error message containing suggestions
            
        Returns:
            List of actionable suggestions
        """
        suggestions = []
        lines = error_message.split('\n')
        
        in_suggestions = False
        for line in lines:
            line = line.strip()
            
            if "Suggestions:" in line or "Setup Instructions:" in line:
                in_suggestions = True
                continue
            
            if in_suggestions:
                if line.startswith("•") or line.startswith("-"):
                    suggestions.append(line[1:].strip())
                elif line.startswith("Option") or line.startswith("Alternative"):
                    suggestions.append(line)
                elif line and not line.startswith("Example") and not line.startswith("Get"):
                    # Stop at next section
                    break
        
        return suggestions