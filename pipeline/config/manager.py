"""
Configuration Manager for Enhanced Transcription & Configuration v0.6.5.

This module handles loading, validation, and merging of configuration from multiple sources:
- System defaults
- User configuration (~/.content-pipeline/config.yaml)
- Project configuration (./.content-pipeline/config.yaml)
- Explicit configuration (--config file.yaml)
- Environment variables
- CLI arguments (highest precedence)

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from .schema import TranscriptionConfig, EngineType, LogLevel, WhisperModelSize
from .environment import EnvironmentVariables
from .yaml_parser import ConfigurationYAMLParser, YAMLParsingError
from .pretty_printer import ConfigurationPrettyPrinter


class ConfigurationManager:
    """Manages configuration loading, validation, and environment variable integration."""
    
    def __init__(self):
        self.user_config_path = Path.home() / ".content-pipeline" / "config.yaml"
        self.project_config_path = Path.cwd() / ".content-pipeline" / "config.yaml"
        self.yaml_parser = ConfigurationYAMLParser()
        self.pretty_printer = ConfigurationPrettyPrinter()
    
    def load_configuration(self, 
                         config_file: Optional[str] = None,
                         cli_overrides: Optional[Dict[str, Any]] = None) -> TranscriptionConfig:
        """
        Load configuration from all sources with proper precedence.
        
        Precedence order (highest to lowest):
        1. CLI arguments (cli_overrides)
        2. Environment variables
        3. Explicit config file (--config)
        4. Project config (./.content-pipeline/config.yaml)
        5. User config (~/.content-pipeline/config.yaml)
        6. System defaults
        
        Args:
            config_file: Optional explicit configuration file path
            cli_overrides: Dictionary of CLI argument overrides
            
        Returns:
            TranscriptionConfig: Merged configuration
            
        Raises:
            ValueError: If configuration files contain invalid YAML or values
        """
        # Start with system defaults
        config_dict = self._get_default_config()
        
        # Layer 1: User configuration
        if self.user_config_path.exists():
            user_config = self._load_yaml_file(self.user_config_path)
            config_dict = self._merge_configs(config_dict, user_config)
        
        # Layer 2: Project configuration
        if self.project_config_path.exists():
            project_config = self._load_yaml_file(self.project_config_path)
            config_dict = self._merge_configs(config_dict, project_config)
        
        # Layer 3: Explicit configuration file
        if config_file:
            explicit_config = self._load_yaml_file(Path(config_file))
            config_dict = self._merge_configs(config_dict, explicit_config)
        
        # Layer 4: Environment variables
        env_overrides = self._load_environment_variables()
        config_dict = self._merge_configs(config_dict, env_overrides)
        
        # Layer 5: CLI overrides (highest precedence)
        if cli_overrides:
            config_dict = self._merge_configs(config_dict, cli_overrides)
        
        # Substitute environment variables in values
        config_dict = self.substitute_environment_variables(config_dict)
        
        # Convert to TranscriptionConfig object
        try:
            config = self._dict_to_config(config_dict)
        except Exception as e:
            raise ValueError(f"Failed to create configuration object: {e}")
        
        return config
    
    def validate_configuration(self, config: TranscriptionConfig) -> List[str]:
        """Validate configuration and return any errors."""
        return config.validate()
    
    def substitute_environment_variables(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute ${VAR} and ${VAR:-default} syntax with environment variable values.
        
        Supports:
        - ${VAR}: Replace with environment variable value
        - ${VAR:-default}: Replace with environment variable value or default if not set
        
        Args:
            config_dict: Configuration dictionary with potential variable references
            
        Returns:
            Dict with environment variables substituted
            
        Raises:
            ValueError: If required environment variable is missing
        """
        def substitute_value(value):
            if not isinstance(value, str):
                return value
            
            # Pattern to match ${VAR} or ${VAR:-default}
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_expr = match.group(1)
                
                # Check for default value syntax
                if ':-' in var_expr:
                    var_name, default_value = var_expr.split(':-', 1)
                    return os.environ.get(var_name, default_value)
                else:
                    var_name = var_expr
                    if var_name not in os.environ:
                        raise ValueError(f"Required environment variable '{var_name}' is not set")
                    return os.environ[var_name]
            
            return re.sub(pattern, replace_var, value)
        
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            else:
                return substitute_value(obj)
        
        return substitute_recursive(config_dict)
    
    def save_configuration(self, config: TranscriptionConfig, file_path: str) -> None:
        """Save configuration to YAML file with comments."""
        config_dict = asdict(config)
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate YAML with comments
        yaml_content = self._generate_commented_yaml(config_dict)
        
        with open(file_path, 'w') as f:
            f.write(yaml_content)
    
    def generate_default_config(self) -> str:
        """Generate default configuration YAML with comments and examples."""
        return self.pretty_printer.generate_full_template(include_examples=True)
    
    def generate_minimal_config(self) -> str:
        """Generate minimal configuration template."""
        return self.pretty_printer.generate_minimal_template()
    
    def generate_engine_config(self, engine_type: str) -> str:
        """Generate configuration template optimized for specific engine."""
        return self.pretty_printer.generate_engine_specific_template(engine_type)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get system default configuration values."""
        return {
            'engine': EngineType.AUTO.value,
            'output_dir': './transcripts',
            'log_level': LogLevel.INFO.value,
            'language': None,
            'whisper_local': {
                'model': WhisperModelSize.BASE.value,
                'device': 'auto',
                'compute_type': 'default',
                'timeout': 300,
                'retry_attempts': 3,
                'retry_delay': 1.0
            },
            'whisper_api': {
                'api_key': None,
                'model': 'whisper-1',
                'temperature': 0.0,
                'response_format': 'json',
                'timeout': 60,
                'retry_attempts': 3,
                'retry_delay': 1.0
            },
            'aws_transcribe': {
                'access_key_id': None,
                'secret_access_key': None,
                'region': 'us-east-1',
                'language_code': 'en-US',
                'media_format': 'auto',
                'timeout': 300,
                'retry_attempts': 3,
                'retry_delay': 1.0
            },
            'auto_selection': {
                'prefer_local': True,
                'fallback_enabled': True,
                'priority_order': [
                    EngineType.WHISPER_LOCAL.value,
                    EngineType.AWS_TRANSCRIBE.value,
                    EngineType.WHISPER_API.value
                ]
            }
        }
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file with enhanced error reporting."""
        try:
            config_dict, validation_errors = self.yaml_parser.validate_and_parse_file(file_path)
            
            # Report validation errors as warnings or errors based on severity
            if validation_errors:
                error_msg = f"Configuration validation errors in {file_path}:\n" + \
                           "\n".join(f"  - {error}" for error in validation_errors)
                raise ValueError(error_msg)
            
            return config_dict
            
        except YAMLParsingError as e:
            # Re-raise with original error message (already includes file path and line info)
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Failed to load configuration file {file_path}: {e}")
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        env_vars = EnvironmentVariables
        
        # Core settings
        if env_vars.DEFAULT_ENGINE in os.environ:
            env_config['engine'] = os.environ[env_vars.DEFAULT_ENGINE]
        
        if env_vars.OUTPUT_DIR in os.environ:
            env_config['output_dir'] = os.environ[env_vars.OUTPUT_DIR]
        
        if env_vars.LOG_LEVEL in os.environ:
            env_config['log_level'] = os.environ[env_vars.LOG_LEVEL]
        
        # Whisper Local configuration
        if env_vars.WHISPER_LOCAL_MODEL in os.environ:
            env_config.setdefault('whisper_local', {})['model'] = os.environ[env_vars.WHISPER_LOCAL_MODEL]
        
        # OpenAI Whisper API configuration
        if env_vars.OPENAI_API_KEY in os.environ:
            env_config.setdefault('whisper_api', {})['api_key'] = os.environ[env_vars.OPENAI_API_KEY]
        
        if env_vars.WHISPER_API_MODEL in os.environ:
            env_config.setdefault('whisper_api', {})['model'] = os.environ[env_vars.WHISPER_API_MODEL]
        
        # AWS credentials and configuration
        if env_vars.AWS_ACCESS_KEY_ID in os.environ:
            env_config.setdefault('aws_transcribe', {})['access_key_id'] = os.environ[env_vars.AWS_ACCESS_KEY_ID]
        
        if env_vars.AWS_SECRET_ACCESS_KEY in os.environ:
            env_config.setdefault('aws_transcribe', {})['secret_access_key'] = os.environ[env_vars.AWS_SECRET_ACCESS_KEY]
        
        if env_vars.AWS_DEFAULT_REGION in os.environ:
            env_config.setdefault('aws_transcribe', {})['region'] = os.environ[env_vars.AWS_DEFAULT_REGION]
        
        return env_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TranscriptionConfig:
        """Convert configuration dictionary to TranscriptionConfig object."""
        # Extract nested configurations
        whisper_local_dict = config_dict.get('whisper_local', {})
        whisper_api_dict = config_dict.get('whisper_api', {})
        aws_transcribe_dict = config_dict.get('aws_transcribe', {})
        auto_selection_dict = config_dict.get('auto_selection', {})
        
        # Import here to avoid circular imports
        from .schema import WhisperLocalConfig, WhisperAPIConfig, AWSTranscribeConfig, AutoSelectionConfig
        
        return TranscriptionConfig(
            engine=config_dict.get('engine', EngineType.AUTO.value),
            output_dir=config_dict.get('output_dir', './transcripts'),
            log_level=config_dict.get('log_level', LogLevel.INFO.value),
            language=config_dict.get('language'),
            whisper_local=WhisperLocalConfig(**whisper_local_dict),
            whisper_api=WhisperAPIConfig(**whisper_api_dict),
            aws_transcribe=AWSTranscribeConfig(**aws_transcribe_dict),
            auto_selection=AutoSelectionConfig(**auto_selection_dict)
        )
    
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