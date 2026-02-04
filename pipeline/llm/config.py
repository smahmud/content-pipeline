"""
LLM Configuration Management

This module provides centralized configuration loading for all LLM providers.
Configuration is loaded with the following precedence:
1. Environment variables (highest priority)
2. Config file values (.content-pipeline/config.yaml)
3. Default values (lowest priority)

All providers accept configuration objects instead of hardcoded values,
enabling flexible deployment across different environments.

Usage:
    >>> from pipeline.llm.config import LLMConfig
    >>> 
    >>> # Load from config file (when llm section exists)
    >>> llm_config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')
    >>> 
    >>> # Or create directly with defaults
    >>> llm_config = LLMConfig()
    >>> 
    >>> # Access provider-specific config
    >>> print(llm_config.ollama.base_url)
    >>> print(llm_config.openai.api_key)
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import os
import yaml
from pathlib import Path


@dataclass
class OllamaConfig:
    """Configuration for Ollama provider.
    
    Attributes:
        base_url: Base URL for Ollama API (default: http://localhost:11434)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        timeout: Request timeout in seconds (local models may be slower)
    """
    base_url: str = "http://localhost:11434"
    default_model: str = "llama2"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI provider.
    
    Attributes:
        api_key: OpenAI API key (required for cloud-openai provider)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        timeout: Request timeout in seconds
        pricing_override: Optional pricing override for enterprise customers
            Format: {"model_name": {"input_per_1k": float, "output_per_1k": float}}
    """
    api_key: str = ""
    default_model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    pricing_override: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock provider.
    
    Attributes:
        region: AWS region to use
        access_key_id: AWS access key ID (optional if using IAM roles)
        secret_access_key: AWS secret access key (optional if using IAM roles)
        session_token: AWS session token (optional, for temporary credentials)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        pricing_override: Optional pricing override for enterprise customers
            Format: {"model_name": {"input_per_1k": float, "output_per_1k": float}}
    """
    region: str = "us-east-1"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    default_model: str = "amazon.nova-lite-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.7
    pricing_override: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic provider.
    
    Attributes:
        api_key: Anthropic API key (required for cloud-anthropic provider)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        timeout: Request timeout in seconds
        pricing_override: Optional pricing override for enterprise customers
            Format: {"model_name": {"input": float, "output": float}} (per 1M tokens)
    """
    api_key: str = ""
    default_model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    pricing_override: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class LLMConfig:
    """Complete LLM configuration for all providers.
    
    This class aggregates configuration for all LLM providers and provides
    a centralized method to load configuration from the config manager.
    
    Attributes:
        ollama: Configuration for Ollama provider
        openai: Configuration for OpenAI provider
        bedrock: Configuration for AWS Bedrock provider
        anthropic: Configuration for Anthropic provider
    """
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    
    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None) -> 'LLMConfig':
        """Load LLM configuration from YAML file.
        
        Configuration precedence (highest to lowest):
        1. Environment variables
        2. Config file values
        3. Default values
        
        Args:
            config_path: Path to config YAML file (default: .content-pipeline/config.yaml)
            
        Returns:
            LLMConfig instance with all provider configurations loaded
            
        Example:
            >>> llm_config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')
        """
        # Load config file if it exists
        llm_section = {}
        if config_path is None:
            config_path = '.content-pipeline/config.yaml'
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                llm_section = config_data.get('llm', {})
        
        return cls.load_from_dict(llm_section)
    
    @classmethod
    def load_from_dict(cls, llm_section: Dict[str, Any]) -> 'LLMConfig':
        """Load LLM configuration from dictionary.
        
        Configuration precedence (highest to lowest):
        1. Environment variables
        2. Config dict values
        3. Default values
        
        Args:
            llm_section: Dictionary containing llm configuration
            
        Returns:
            LLMConfig instance with all provider configurations loaded
            
        Example:
            >>> config_dict = {'ollama': {'base_url': 'http://localhost:11434'}}
            >>> llm_config = LLMConfig.load_from_dict(config_dict)
        """
        
        # Load Ollama config
        ollama_config = OllamaConfig(
            base_url=cls._resolve_value(
                llm_section.get('ollama', {}).get('base_url'),
                'OLLAMA_BASE_URL',
                'http://localhost:11434'
            ),
            default_model=cls._resolve_value(
                llm_section.get('ollama', {}).get('default_model'),
                'OLLAMA_MODEL',
                'llama2'
            ),
            max_tokens=int(cls._resolve_value(
                llm_section.get('ollama', {}).get('max_tokens'),
                'OLLAMA_MAX_TOKENS',
                4096
            )),
            temperature=float(cls._resolve_value(
                llm_section.get('ollama', {}).get('temperature'),
                'OLLAMA_TEMPERATURE',
                0.3
            )),
            timeout=int(cls._resolve_value(
                llm_section.get('ollama', {}).get('timeout'),
                'OLLAMA_TIMEOUT',
                120
            ))
        )
        
        # Load OpenAI config
        openai_config = OpenAIConfig(
            api_key=cls._resolve_value(
                llm_section.get('openai', {}).get('api_key'),
                'OPENAI_API_KEY',
                ''
            ),
            default_model=cls._resolve_value(
                llm_section.get('openai', {}).get('default_model'),
                'OPENAI_MODEL',
                'gpt-4'
            ),
            max_tokens=int(cls._resolve_value(
                llm_section.get('openai', {}).get('max_tokens'),
                'OPENAI_MAX_TOKENS',
                4096
            )),
            temperature=float(cls._resolve_value(
                llm_section.get('openai', {}).get('temperature'),
                'OPENAI_TEMPERATURE',
                0.7
            )),
            timeout=int(cls._resolve_value(
                llm_section.get('openai', {}).get('timeout'),
                'OPENAI_TIMEOUT',
                60
            ))
        )
        
        # Load Bedrock config
        bedrock_config = BedrockConfig(
            region=cls._resolve_value(
                llm_section.get('bedrock', {}).get('region'),
                'AWS_REGION',
                'us-east-1'
            ),
            access_key_id=cls._resolve_value(
                llm_section.get('bedrock', {}).get('access_key_id'),
                'AWS_ACCESS_KEY_ID',
                None
            ),
            secret_access_key=cls._resolve_value(
                llm_section.get('bedrock', {}).get('secret_access_key'),
                'AWS_SECRET_ACCESS_KEY',
                None
            ),
            session_token=cls._resolve_value(
                llm_section.get('bedrock', {}).get('session_token'),
                'AWS_SESSION_TOKEN',
                None
            ),
            default_model=cls._resolve_value(
                llm_section.get('bedrock', {}).get('default_model'),
                'BEDROCK_MODEL',
                'amazon.nova-lite-v1:0'
            ),
            max_tokens=int(cls._resolve_value(
                llm_section.get('bedrock', {}).get('max_tokens'),
                'BEDROCK_MAX_TOKENS',
                4096
            )),
            temperature=float(cls._resolve_value(
                llm_section.get('bedrock', {}).get('temperature'),
                'BEDROCK_TEMPERATURE',
                0.7
            ))
        )
        
        # Load Anthropic config
        anthropic_config = AnthropicConfig(
            api_key=cls._resolve_value(
                llm_section.get('anthropic', {}).get('api_key'),
                'ANTHROPIC_API_KEY',
                ''
            ),
            default_model=cls._resolve_value(
                llm_section.get('anthropic', {}).get('default_model'),
                'ANTHROPIC_MODEL',
                'claude-3-opus-20240229'
            ),
            max_tokens=int(cls._resolve_value(
                llm_section.get('anthropic', {}).get('max_tokens'),
                'ANTHROPIC_MAX_TOKENS',
                4096
            )),
            temperature=float(cls._resolve_value(
                llm_section.get('anthropic', {}).get('temperature'),
                'ANTHROPIC_TEMPERATURE',
                0.7
            )),
            timeout=int(cls._resolve_value(
                llm_section.get('anthropic', {}).get('timeout'),
                'ANTHROPIC_TIMEOUT',
                60
            ))
        )
        
        return cls(
            ollama=ollama_config,
            openai=openai_config,
            bedrock=bedrock_config,
            anthropic=anthropic_config
        )
    
    @staticmethod
    def _resolve_value(config_value: Any, env_var: str, default: Any) -> Any:
        """Resolve configuration value with precedence: ENV > Config > Default.
        
        This method implements the configuration precedence rules:
        1. If environment variable is set, use it (highest priority)
        2. If config file has a value, use it (medium priority)
        3. Otherwise, use the default value (lowest priority)
        
        Args:
            config_value: Value from config file (may be None)
            env_var: Environment variable name to check
            default: Default value to use if neither env nor config is set
            
        Returns:
            Resolved configuration value
            
        Example:
            >>> # With OLLAMA_BASE_URL="http://remote:11434" in environment
            >>> _resolve_value(None, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            'http://remote:11434'
            >>> 
            >>> # Without environment variable, with config value
            >>> _resolve_value('http://config:11434', 'OLLAMA_BASE_URL', 'http://localhost:11434')
            'http://config:11434'
            >>> 
            >>> # Without environment variable or config value
            >>> _resolve_value(None, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            'http://localhost:11434'
        """
        # Check environment variable first (highest priority)
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        
        # Check config file value (medium priority)
        if config_value is not None:
            return config_value
        
        # Use default (lowest priority)
        return default
