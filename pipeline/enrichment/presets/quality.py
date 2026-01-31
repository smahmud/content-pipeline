"""
Quality Presets

Defines quality presets for enrichment operations that balance
output quality with cost and speed. Each preset selects appropriate
models for each provider.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class QualityLevel(str, Enum):
    """Quality level enumeration."""
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


@dataclass
class QualityPreset:
    """Quality preset configuration.
    
    Attributes:
        name: Preset name (fast, balanced, best)
        description: Human-readable description
        model_selections: Dict mapping provider to model name
        temperature: Temperature setting for generation
        max_tokens_multiplier: Multiplier for max_tokens (1.0 = default)
    """
    name: str
    description: str
    model_selections: Dict[str, str]
    temperature: float = 0.3
    max_tokens_multiplier: float = 1.0


class QualityPresets:
    """Quality preset definitions for all providers.
    
    This class defines three quality levels:
    - FAST: Smaller, cheaper models for quick results
    - BALANCED: Mid-tier models balancing quality and cost (default)
    - BEST: Largest, highest-quality models
    
    Example:
        >>> preset = QualityPresets.BALANCED
        >>> model = preset.model_selections["openai"]
        >>> # Returns "gpt-4-turbo"
    """
    
    FAST = QualityPreset(
        name="fast",
        description="Fast processing with smaller models (lower cost, good quality)",
        model_selections={
            "openai": "gpt-3.5-turbo",
            "claude": "claude-3-haiku-20240307",
            "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
            "ollama": "llama2:7b"
        },
        temperature=0.3,
        max_tokens_multiplier=0.8  # Slightly fewer tokens for speed
    )
    
    BALANCED = QualityPreset(
        name="balanced",
        description="Balanced quality and cost with mid-tier models (default)",
        model_selections={
            "openai": "gpt-4-turbo",
            "claude": "claude-3-sonnet-20240229",
            "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
            "ollama": "llama2:13b"
        },
        temperature=0.3,
        max_tokens_multiplier=1.0  # Default token allocation
    )
    
    BEST = QualityPreset(
        name="best",
        description="Highest quality with largest models (higher cost, best results)",
        model_selections={
            "openai": "gpt-4",
            "claude": "claude-3-opus-20240229",
            "bedrock": "anthropic.claude-3-opus-20240229-v1:0",
            "ollama": "llama2:70b"
        },
        temperature=0.2,  # Lower temperature for more focused output
        max_tokens_multiplier=1.2  # More tokens for detailed output
    )
    
    @classmethod
    def get_preset(cls, name: str) -> QualityPreset:
        """Get preset by name.
        
        Args:
            name: Preset name (fast, balanced, best)
            
        Returns:
            Quality preset
            
        Raises:
            ValueError: If preset name is invalid
        """
        presets = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "best": cls.BEST
        }
        
        if name not in presets:
            raise ValueError(
                f"Invalid quality preset: {name}. "
                f"Valid options: {', '.join(presets.keys())}"
            )
        
        return presets[name]
    
    @classmethod
    def get_model_for_provider(
        cls,
        quality: str,
        provider: str
    ) -> str:
        """Get model for specific provider and quality level.
        
        Args:
            quality: Quality level (fast, balanced, best)
            provider: Provider name (openai, claude, bedrock, ollama)
            
        Returns:
            Model identifier for the provider
            
        Raises:
            ValueError: If quality or provider is invalid
        """
        preset = cls.get_preset(quality)
        
        if provider not in preset.model_selections:
            raise ValueError(
                f"Provider '{provider}' not supported in quality presets. "
                f"Supported providers: {', '.join(preset.model_selections.keys())}"
            )
        
        return preset.model_selections[provider]
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """List all available presets with descriptions.
        
        Returns:
            Dict mapping preset name to description
        """
        return {
            "fast": cls.FAST.description,
            "balanced": cls.BALANCED.description,
            "best": cls.BEST.description
        }


def get_quality_preset(name: str) -> QualityPreset:
    """Convenience function to get quality preset.
    
    Args:
        name: Preset name (fast, balanced, best)
        
    Returns:
        Quality preset
    """
    return QualityPresets.get_preset(name)


def apply_quality_preset(
    preset: QualityPreset,
    provider: str,
    base_max_tokens: int
) -> Dict[str, any]:
    """Apply quality preset to get model and parameters.
    
    Args:
        preset: Quality preset to apply
        provider: Provider name
        base_max_tokens: Base max_tokens value
        
    Returns:
        Dict with model, temperature, and max_tokens
    """
    return {
        "model": preset.model_selections.get(provider),
        "temperature": preset.temperature,
        "max_tokens": int(base_max_tokens * preset.max_tokens_multiplier)
    }
