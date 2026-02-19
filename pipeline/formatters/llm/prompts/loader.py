"""
Prompt Loader

Loads enhancement prompts from YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml


class PromptLoader:
    """Loads enhancement prompts from YAML files.
    
    Example:
        >>> loader = PromptLoader()
        >>> prompt = loader.get_prompt("blog", tone="professional", length="medium")
    """
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).parent)
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, dict] = {}
    
    def get_prompt(
        self,
        output_type: str,
        tone: str = "professional",
        length: str = "medium",
        content: str = "",
    ) -> str:
        """Get enhancement prompt for output type.
        
        Args:
            output_type: Output type (blog, tweet, etc.)
            tone: Writing tone
            length: Output length preference
            content: Content to enhance
            
        Returns:
            Rendered prompt string
        """
        prompt_data = self._load_prompt(output_type)
        
        if prompt_data is None:
            # Return generic prompt
            return self._build_generic_prompt(output_type, tone, length, content)
        
        # Get tone and length guidance
        tone_guidance = prompt_data.get("tones", {}).get(tone, "")
        length_guidance = prompt_data.get("lengths", {}).get(length, "")
        
        # Build prompt from template
        template = prompt_data.get("template", "")
        
        return template.format(
            output_type=output_type,
            tone=tone,
            tone_guidance=tone_guidance,
            length=length,
            length_guidance=length_guidance,
            content=content,
            rules=prompt_data.get("rules", ""),
        )
    
    def _load_prompt(self, output_type: str) -> Optional[dict]:
        """Load prompt data from YAML file.
        
        Args:
            output_type: Output type
            
        Returns:
            Prompt data dict or None if not found
        """
        if output_type in self._cache:
            return self._cache[output_type]
        
        # Try to load from file
        filename = f"enhance_{output_type.replace('-', '_')}.yaml"
        filepath = self.prompts_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self._cache[output_type] = data
                return data
        except Exception:
            return None
    
    def _build_generic_prompt(
        self,
        output_type: str,
        tone: str,
        length: str,
        content: str,
    ) -> str:
        """Build generic enhancement prompt.
        
        Args:
            output_type: Output type
            tone: Writing tone
            length: Output length
            content: Content to enhance
            
        Returns:
            Generic prompt string
        """
        return f"""You are an expert content editor. Enhance the following {output_type} content.

TONE: {tone}
LENGTH: {length}

RULES:
1. Preserve all structural elements (headers, lists, code blocks)
2. Improve prose quality and engagement
3. Maintain original meaning and information
4. Do not add or remove sections

CONTENT:
{content}

ENHANCED CONTENT:"""
    
    def list_available_prompts(self) -> list:
        """List available prompt files.
        
        Returns:
            List of output types with custom prompts
        """
        prompts = []
        for filepath in self.prompts_dir.glob("enhance_*.yaml"):
            output_type = filepath.stem.replace("enhance_", "").replace("_", "-")
            prompts.append(output_type)
        return prompts
