"""
Prompt Renderer

Renders prompt templates with transcript context using Jinja2.
Supports variable substitution and conditional logic in templates.
"""

from jinja2 import Template, TemplateError, Environment, StrictUndefined
from typing import Dict, Any, Optional

from pipeline.enrichment.errors import PromptRenderError


class PromptRenderer:
    """Renders prompt templates with transcript context.
    
    Uses Jinja2 templating engine to render prompts with dynamic
    context variables from transcripts and additional metadata.
    
    Attributes:
        strict_mode: If True, raises error for undefined variables
    """
    
    def __init__(self, strict_mode: bool = True):
        """Initialize the prompt renderer.
        
        Args:
            strict_mode: If True, undefined variables raise errors
        """
        self.strict_mode = strict_mode
        
        # Create Jinja2 environment
        if strict_mode:
            self.env = Environment(undefined=StrictUndefined)
        else:
            self.env = Environment()
    
    def render(
        self,
        prompt_template: Dict[str, Any],
        transcript_text: str,
        transcript_language: str = "en",
        transcript_duration: str = "00:00:00",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render prompt template with context.
        
        Combines system and user prompts with transcript context variables.
        
        Args:
            prompt_template: Template dict with 'system' and 'user_template' keys
            transcript_text: Full transcript text
            transcript_language: Language code (e.g., "en")
            transcript_duration: Duration string (e.g., "01:23:45")
            additional_context: Optional additional context variables
            
        Returns:
            Rendered prompt string (system + user combined)
            
        Raises:
            PromptRenderError: If template rendering fails
        """
        # Build context with standard variables
        context = {
            "transcript_text": transcript_text,
            "transcript_language": transcript_language,
            "transcript_duration": transcript_duration,
            "word_count": len(transcript_text.split()),
        }
        
        # Add any additional context
        if additional_context:
            context.update(additional_context)
        
        try:
            # Render system prompt
            system_template = self.env.from_string(prompt_template["system"])
            system_prompt = system_template.render(**context)
            
            # Render user prompt
            user_template = self.env.from_string(prompt_template["user_template"])
            user_prompt = user_template.render(**context)
            
            # Combine into single prompt
            # Most LLMs expect system message followed by user message
            return f"{system_prompt}\n\n{user_prompt}"
        
        except TemplateError as e:
            raise PromptRenderError(
                f"Error rendering prompt template: {e}"
            )
        except KeyError as e:
            raise PromptRenderError(
                f"Prompt template missing required field: {e}"
            )
        except Exception as e:
            raise PromptRenderError(
                f"Unexpected error rendering prompt: {e}"
            )
    
    def render_system_only(
        self,
        prompt_template: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render only the system prompt.
        
        Useful for testing or when system prompt doesn't need transcript context.
        
        Args:
            prompt_template: Template dict with 'system' key
            context: Optional context variables
            
        Returns:
            Rendered system prompt
            
        Raises:
            PromptRenderError: If rendering fails
        """
        try:
            system_template = self.env.from_string(prompt_template["system"])
            return system_template.render(**(context or {}))
        except Exception as e:
            raise PromptRenderError(
                f"Error rendering system prompt: {e}"
            )
    
    def render_user_only(
        self,
        prompt_template: Dict[str, Any],
        transcript_text: str,
        transcript_language: str = "en",
        transcript_duration: str = "00:00:00",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render only the user prompt.
        
        Useful for testing or when system prompt is handled separately.
        
        Args:
            prompt_template: Template dict with 'user_template' key
            transcript_text: Full transcript text
            transcript_language: Language code
            transcript_duration: Duration string
            additional_context: Optional additional context
            
        Returns:
            Rendered user prompt
            
        Raises:
            PromptRenderError: If rendering fails
        """
        context = {
            "transcript_text": transcript_text,
            "transcript_language": transcript_language,
            "transcript_duration": transcript_duration,
            "word_count": len(transcript_text.split()),
        }
        
        if additional_context:
            context.update(additional_context)
        
        try:
            user_template = self.env.from_string(prompt_template["user_template"])
            return user_template.render(**context)
        except Exception as e:
            raise PromptRenderError(
                f"Error rendering user prompt: {e}"
            )
    
    def validate_template(
        self,
        template_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate that a template string can be rendered.
        
        Useful for testing templates before use.
        
        Args:
            template_string: Jinja2 template string
            context: Optional context for rendering
            
        Returns:
            True if template is valid
            
        Raises:
            PromptRenderError: If template is invalid
        """
        try:
            template = self.env.from_string(template_string)
            template.render(**(context or {}))
            return True
        except Exception as e:
            raise PromptRenderError(
                f"Template validation failed: {e}"
            )


def render_prompt(
    prompt_template: Dict[str, Any],
    transcript_text: str,
    transcript_language: str = "en",
    transcript_duration: str = "00:00:00",
    additional_context: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True
) -> str:
    """Convenience function to render a prompt.
    
    Args:
        prompt_template: Template dict with 'system' and 'user_template'
        transcript_text: Full transcript text
        transcript_language: Language code
        transcript_duration: Duration string
        additional_context: Optional additional context
        strict_mode: If True, undefined variables raise errors
        
    Returns:
        Rendered prompt string
        
    Raises:
        PromptRenderError: If rendering fails
    """
    renderer = PromptRenderer(strict_mode=strict_mode)
    return renderer.render(
        prompt_template,
        transcript_text,
        transcript_language,
        transcript_duration,
        additional_context
    )
