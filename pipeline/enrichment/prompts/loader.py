"""
Prompt Loader

Loads and validates YAML prompt templates for enrichment operations.
Supports custom prompt directories with fallback to defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.enrichment.errors import PromptTemplateError


class PromptLoader:
    """Loads and validates YAML prompt templates.
    
    Supports loading prompts from custom directories with automatic
    fallback to default templates. Caches loaded prompts for performance.
    
    Attributes:
        default_prompts_dir: Directory containing default prompt templates
        custom_prompts_dir: Optional directory for custom templates
    """
    
    def __init__(
        self,
        default_prompts_dir: Optional[Path] = None,
        custom_prompts_dir: Optional[Path] = None
    ):
        """Initialize the prompt loader.
        
        Args:
            default_prompts_dir: Directory with default prompts (defaults to package prompts/)
            custom_prompts_dir: Optional directory with custom prompts
        """
        if default_prompts_dir is None:
            # Use the prompts directory in the package
            default_prompts_dir = Path(__file__).parent
        
        self.default_prompts_dir = Path(default_prompts_dir)
        self.custom_prompts_dir = Path(custom_prompts_dir) if custom_prompts_dir else None
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}
        
        # Validate default prompts directory exists
        if not self.default_prompts_dir.exists():
            raise PromptTemplateError(
                f"Default prompts directory does not exist: {self.default_prompts_dir}"
            )
    
    def load_prompt(self, enrichment_type: str) -> Dict[str, Any]:
        """Load prompt template for enrichment type.
        
        Checks custom prompts directory first, then falls back to defaults.
        Caches loaded prompts for performance.
        
        Args:
            enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
            
        Returns:
            Prompt template dictionary with 'system', 'user_template', and 'expected_output'
            
        Raises:
            PromptTemplateError: If prompt cannot be loaded or is invalid
        """
        # Check cache first
        if enrichment_type in self._prompt_cache:
            return self._prompt_cache[enrichment_type]
        
        # Map enrichment types to template filenames
        template_map = {
            'summary': 'summarize.yaml',
            'tag': 'tag.yaml',
            'chapter': 'chapterize.yaml',
            'highlight': 'highlight.yaml',
        }
        
        if enrichment_type not in template_map:
            raise PromptTemplateError(
                f"Unknown enrichment type: {enrichment_type}. "
                f"Must be one of: {list(template_map.keys())}"
            )
        
        template_filename = template_map[enrichment_type]
        
        # Check custom prompts first
        if self.custom_prompts_dir:
            custom_path = self.custom_prompts_dir / template_filename
            if custom_path.exists():
                try:
                    prompt = self._load_yaml(custom_path)
                    self._validate_prompt(prompt, enrichment_type)
                    self._prompt_cache[enrichment_type] = prompt
                    return prompt
                except Exception as e:
                    raise PromptTemplateError(
                        f"Error loading custom prompt from {custom_path}: {e}"
                    )
        
        # Fall back to default prompts
        default_path = self.default_prompts_dir / template_filename
        if not default_path.exists():
            raise PromptTemplateError(
                f"No prompt template found for {enrichment_type} at {default_path}"
            )
        
        try:
            prompt = self._load_yaml(default_path)
            self._validate_prompt(prompt, enrichment_type)
            self._prompt_cache[enrichment_type] = prompt
            return prompt
        except Exception as e:
            raise PromptTemplateError(
                f"Error loading default prompt from {default_path}: {e}"
            )
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Parsed YAML data
            
        Raises:
            PromptTemplateError: If YAML is invalid
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    raise PromptTemplateError(
                        f"YAML file must contain a dictionary, got {type(data).__name__}"
                    )
                return data
        except yaml.YAMLError as e:
            raise PromptTemplateError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise PromptTemplateError(f"Error reading {path}: {e}")
    
    def _validate_prompt(self, prompt: Dict[str, Any], enrichment_type: str):
        """Validate prompt structure.
        
        Ensures prompt has all required fields and proper structure.
        
        Args:
            prompt: Prompt template dictionary
            enrichment_type: Type of enrichment
            
        Raises:
            PromptTemplateError: If prompt is invalid
        """
        required_fields = ["system", "user_template", "expected_output"]
        
        for field in required_fields:
            if field not in prompt:
                raise PromptTemplateError(
                    f"Prompt for {enrichment_type} missing required field: {field}"
                )
        
        # Validate field types
        if not isinstance(prompt["system"], str):
            raise PromptTemplateError(
                f"Prompt 'system' field must be a string, got {type(prompt['system']).__name__}"
            )
        
        if not isinstance(prompt["user_template"], str):
            raise PromptTemplateError(
                f"Prompt 'user_template' field must be a string, got {type(prompt['user_template']).__name__}"
            )
        
        if not isinstance(prompt["expected_output"], dict):
            raise PromptTemplateError(
                f"Prompt 'expected_output' field must be a dictionary, got {type(prompt['expected_output']).__name__}"
            )
        
        # Validate that templates are not empty
        if not prompt["system"].strip():
            raise PromptTemplateError(
                f"Prompt 'system' field cannot be empty for {enrichment_type}"
            )
        
        if not prompt["user_template"].strip():
            raise PromptTemplateError(
                f"Prompt 'user_template' field cannot be empty for {enrichment_type}"
            )
    
    def clear_cache(self):
        """Clear the prompt cache.
        
        Useful when prompts have been modified and need to be reloaded.
        """
        self._prompt_cache.clear()
    
    def get_available_enrichment_types(self) -> list[str]:
        """Get list of available enrichment types.
        
        Returns:
            List of enrichment type names
        """
        return ['summary', 'tag', 'chapter', 'highlight']


def load_prompt(
    enrichment_type: str,
    custom_prompts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Convenience function to load a prompt template.
    
    Args:
        enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
        custom_prompts_dir: Optional directory with custom prompts
        
    Returns:
        Prompt template dictionary
        
    Raises:
        PromptTemplateError: If prompt cannot be loaded
    """
    loader = PromptLoader(custom_prompts_dir=custom_prompts_dir)
    return loader.load_prompt(enrichment_type)
