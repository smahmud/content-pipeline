"""
Enhanced YAML parser with validation for Enhanced Transcription & Configuration v0.6.5.

This module provides YAML parsing with detailed error reporting, line number information,
and schema validation for configuration files.

Requirements: 5.6, 5.7, 11.1, 11.2, 11.3
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

from .schema import TranscriptionConfig


class YAMLParsingError(Exception):
    """Custom exception for YAML parsing errors with enhanced context."""
    
    def __init__(self, message: str, file_path: Optional[Path] = None, 
                 line_number: Optional[int] = None, column: Optional[int] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.column = column
        
        # Build detailed error message
        error_parts = [message]
        
        if file_path:
            error_parts.append(f"File: {file_path}")
        
        if line_number is not None:
            if column is not None:
                error_parts.append(f"Line {line_number}, Column {column}")
            else:
                error_parts.append(f"Line {line_number}")
        
        super().__init__(" | ".join(error_parts))


class ConfigurationYAMLParser:
    """Enhanced YAML parser for configuration files with validation and error reporting."""
    
    def __init__(self):
        self.loader = yaml.SafeLoader
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse YAML configuration file with enhanced error reporting.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing parsed configuration
            
        Raises:
            YAMLParsingError: If YAML is invalid or file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content or {}
                
        except yaml.YAMLError as e:
            # Extract line and column information if available
            line_number = None
            column = None
            
            if hasattr(e, 'problem_mark') and e.problem_mark:
                line_number = e.problem_mark.line + 1  # YAML uses 0-based line numbers
                column = e.problem_mark.column + 1
            
            # Create descriptive error message
            if hasattr(e, 'problem') and e.problem:
                message = f"YAML parsing error: {e.problem}"
            else:
                message = f"YAML parsing error: {str(e)}"
            
            raise YAMLParsingError(message, file_path, line_number, column)
            
        except FileNotFoundError:
            raise YAMLParsingError(f"Configuration file not found", file_path)
            
        except PermissionError:
            raise YAMLParsingError(f"Permission denied reading configuration file", file_path)
            
        except UnicodeDecodeError as e:
            raise YAMLParsingError(f"File encoding error: {e}", file_path)
            
        except Exception as e:
            raise YAMLParsingError(f"Unexpected error reading configuration file: {e}", file_path)
    
    def parse_string(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML configuration from string with enhanced error reporting.
        
        Args:
            yaml_content: YAML content as string
            
        Returns:
            Dictionary containing parsed configuration
            
        Raises:
            YAMLParsingError: If YAML is invalid
        """
        try:
            content = yaml.safe_load(yaml_content)
            return content or {}
            
        except yaml.YAMLError as e:
            # Extract line and column information if available
            line_number = None
            column = None
            
            if hasattr(e, 'problem_mark') and e.problem_mark:
                line_number = e.problem_mark.line + 1
                column = e.problem_mark.column + 1
            
            # Create descriptive error message
            if hasattr(e, 'problem') and e.problem:
                message = f"YAML parsing error: {e.problem}"
            else:
                message = f"YAML parsing error: {str(e)}"
            
            raise YAMLParsingError(message, None, line_number, column)
    
    def validate_configuration_structure(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate configuration dictionary structure against expected schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for unknown top-level keys
        expected_keys = {
            'engine', 'output_dir', 'log_level', 'language',
            'whisper_local', 'whisper_api', 'aws_transcribe', 'auto_selection'
        }
        
        unknown_keys = set(config_dict.keys()) - expected_keys
        if unknown_keys:
            errors.append(f"Unknown configuration keys: {', '.join(sorted(unknown_keys))}")
        
        # Validate nested structures
        if 'whisper_local' in config_dict:
            errors.extend(self._validate_whisper_local_config(config_dict['whisper_local']))
        
        if 'whisper_api' in config_dict:
            errors.extend(self._validate_whisper_api_config(config_dict['whisper_api']))
        
        if 'aws_transcribe' in config_dict:
            errors.extend(self._validate_aws_transcribe_config(config_dict['aws_transcribe']))
        
        if 'auto_selection' in config_dict:
            errors.extend(self._validate_auto_selection_config(config_dict['auto_selection']))
        
        return errors
    
    def _validate_whisper_local_config(self, config: Any) -> List[str]:
        """Validate whisper_local configuration section."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("whisper_local must be a dictionary")
            return errors
        
        expected_keys = {'model', 'device', 'compute_type', 'timeout', 'retry_attempts', 'retry_delay'}
        unknown_keys = set(config.keys()) - expected_keys
        if unknown_keys:
            errors.append(f"Unknown whisper_local keys: {', '.join(sorted(unknown_keys))}")
        
        # Validate specific fields
        if 'timeout' in config and not isinstance(config['timeout'], (int, float)):
            errors.append("whisper_local.timeout must be a number")
        
        if 'retry_attempts' in config and not isinstance(config['retry_attempts'], int):
            errors.append("whisper_local.retry_attempts must be an integer")
        
        return errors
    
    def _validate_whisper_api_config(self, config: Any) -> List[str]:
        """Validate whisper_api configuration section."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("whisper_api must be a dictionary")
            return errors
        
        expected_keys = {'api_key', 'model', 'temperature', 'response_format', 'timeout', 'retry_attempts', 'retry_delay'}
        unknown_keys = set(config.keys()) - expected_keys
        if unknown_keys:
            errors.append(f"Unknown whisper_api keys: {', '.join(sorted(unknown_keys))}")
        
        # Validate specific fields
        if 'temperature' in config and not isinstance(config['temperature'], (int, float)):
            errors.append("whisper_api.temperature must be a number")
        
        if 'timeout' in config and not isinstance(config['timeout'], (int, float)):
            errors.append("whisper_api.timeout must be a number")
        
        return errors
    
    def _validate_aws_transcribe_config(self, config: Any) -> List[str]:
        """Validate aws_transcribe configuration section."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("aws_transcribe must be a dictionary")
            return errors
        
        expected_keys = {
            'access_key_id', 'secret_access_key', 'region', 'language_code', 
            's3_bucket', 'media_format', 'timeout', 'retry_attempts', 'retry_delay'
        }
        unknown_keys = set(config.keys()) - expected_keys
        if unknown_keys:
            errors.append(f"Unknown aws_transcribe keys: {', '.join(sorted(unknown_keys))}")
        
        return errors
    
    def _validate_auto_selection_config(self, config: Any) -> List[str]:
        """Validate auto_selection configuration section."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("auto_selection must be a dictionary")
            return errors
        
        expected_keys = {'prefer_local', 'fallback_enabled', 'priority_order'}
        unknown_keys = set(config.keys()) - expected_keys
        if unknown_keys:
            errors.append(f"Unknown auto_selection keys: {', '.join(sorted(unknown_keys))}")
        
        # Validate priority_order
        if 'priority_order' in config:
            priority_order = config['priority_order']
            if not isinstance(priority_order, list):
                errors.append("auto_selection.priority_order must be a list")
            else:
                valid_engines = {'whisper-local', 'whisper-api', 'aws-transcribe'}
                for engine in priority_order:
                    if engine not in valid_engines:
                        errors.append(f"Invalid engine in priority_order: '{engine}'. "
                                    f"Valid options: {', '.join(sorted(valid_engines))}")
        
        return errors
    
    def serialize_to_yaml(self, config: TranscriptionConfig, include_comments: bool = True) -> str:
        """
        Serialize configuration to YAML string with optional comments.
        
        Args:
            config: TranscriptionConfig object to serialize
            include_comments: Whether to include helpful comments
            
        Returns:
            YAML string representation of configuration
        """
        config_dict = asdict(config)
        
        if include_comments:
            return self._generate_commented_yaml(config_dict)
        else:
            return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    
    def _generate_commented_yaml(self, config_dict: Dict[str, Any]) -> str:
        """Generate YAML configuration with helpful comments."""
        return f"""# Content Pipeline Configuration v0.6.5
# Enhanced Transcription & Configuration
# This file configures transcription engines and output settings

# Default transcription engine (whisper-local, whisper-api, aws-transcribe, auto)
engine: {config_dict['engine']}

# Output directory for transcripts (supports environment variables)
output_dir: {config_dict['output_dir']}

# Logging level (debug, info, warning, error)
log_level: {config_dict['log_level']}

# Default language hint for transcription (optional)
language: {config_dict['language']}

# Local Whisper configuration
whisper_local:
  model: {config_dict['whisper_local']['model']}  # tiny, base, small, medium, large
  device: {config_dict['whisper_local']['device']}  # cpu, cuda, auto
  compute_type: {config_dict['whisper_local']['compute_type']}
  timeout: {config_dict['whisper_local']['timeout']}
  retry_attempts: {config_dict['whisper_local']['retry_attempts']}
  retry_delay: {config_dict['whisper_local']['retry_delay']}

# OpenAI Whisper API configuration
whisper_api:
  api_key: ${{OPENAI_API_KEY}}  # Environment variable substitution
  model: {config_dict['whisper_api']['model']}
  temperature: {config_dict['whisper_api']['temperature']}
  response_format: {config_dict['whisper_api']['response_format']}
  timeout: {config_dict['whisper_api']['timeout']}
  retry_attempts: {config_dict['whisper_api']['retry_attempts']}
  retry_delay: {config_dict['whisper_api']['retry_delay']}

# AWS Transcribe configuration
aws_transcribe:
  access_key_id: ${{AWS_ACCESS_KEY_ID}}  # Environment variable substitution
  secret_access_key: ${{AWS_SECRET_ACCESS_KEY}}
  region: {config_dict['aws_transcribe']['region']}
  language_code: {config_dict['aws_transcribe']['language_code']}
  media_format: {config_dict['aws_transcribe']['media_format']}
  timeout: {config_dict['aws_transcribe']['timeout']}
  retry_attempts: {config_dict['aws_transcribe']['retry_attempts']}
  retry_delay: {config_dict['aws_transcribe']['retry_delay']}

# Auto-selection preferences
auto_selection:
  prefer_local: {str(config_dict['auto_selection']['prefer_local']).lower()}
  fallback_enabled: {str(config_dict['auto_selection']['fallback_enabled']).lower()}
  priority_order:
{chr(10).join(f'    - {engine}' for engine in config_dict['auto_selection']['priority_order'])}
"""
    
    def validate_and_parse_file(self, file_path: Path) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parse and validate YAML configuration file in one step.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Tuple of (parsed_config, validation_errors)
            
        Raises:
            YAMLParsingError: If YAML parsing fails
        """
        config_dict = self.parse_file(file_path)
        validation_errors = self.validate_configuration_structure(config_dict)
        return config_dict, validation_errors