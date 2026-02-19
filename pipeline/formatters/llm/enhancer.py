"""
LLM Enhancer

Enhances template-rendered content using LLM providers.
Reuses the agent infrastructure from the enrichment module.

Implements graceful degradation: if LLM enhancement fails after all
retries, falls back to template-only output with a warning.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from pipeline.enrichment.agents.factory import AgentFactory
from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse
from pipeline.enrichment.errors import LLMProviderError
from pipeline.formatters.style_profile import StyleProfile
from pipeline.formatters.retry import (
    retry_enhancement,
    EnhancementRetryContext,
    is_transient_error,
    is_permanent_error,
    TRANSIENT_ERRORS,
    PERMANENT_ERRORS,
)
from pipeline.formatters.errors import EnhancementError


logger = logging.getLogger(__name__)


# Valid tone options
VALID_TONES = ["professional", "casual", "technical", "friendly"]

# Valid length options
VALID_LENGTHS = ["short", "medium", "long"]

# Default enhancement settings
DEFAULT_TONE = "professional"
DEFAULT_LENGTH = "medium"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000


@dataclass
class EnhancementConfig:
    """Configuration for LLM enhancement.
    
    Attributes:
        provider: LLM provider to use ("auto", "cloud-openai", etc.)
        model: Specific model to use (None for provider default)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens for response
        tone: Writing tone (professional, casual, technical, friendly)
        length: Output length preference (short, medium, long)
    """
    provider: str = "auto"
    model: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    tone: str = DEFAULT_TONE
    length: str = DEFAULT_LENGTH
    
    def __post_init__(self):
        """Validate configuration."""
        if self.tone not in VALID_TONES:
            raise ValueError(f"Invalid tone: {self.tone}. Valid: {VALID_TONES}")
        if self.length not in VALID_LENGTHS:
            raise ValueError(f"Invalid length: {self.length}. Valid: {VALID_LENGTHS}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive")


@dataclass
class EnhancementResult:
    """Result of LLM enhancement.
    
    Attributes:
        content: Enhanced content
        success: Whether enhancement succeeded
        provider: Provider used
        model: Model used
        tokens_used: Total tokens consumed
        cost_usd: Cost in USD
        enhanced: Whether content was actually enhanced (vs fallback)
        error: Error message if failed
        warnings: Any warnings generated
    """
    content: str
    success: bool
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    enhanced: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class LLMEnhancer:
    """Enhances template output using LLM providers.
    
    This class provides LLM-powered prose enhancement for formatted content.
    It reuses the agent infrastructure from the enrichment module and supports
    style profiles for customization.
    
    Example:
        >>> enhancer = LLMEnhancer()
        >>> result = enhancer.enhance(
        ...     content="# Blog Post\\n\\nThis is content.",
        ...     output_type="blog",
        ...     tone="professional",
        ...     length="medium"
        ... )
    """
    
    def __init__(
        self,
        agent_factory: Optional[AgentFactory] = None,
        default_provider: str = "auto"
    ):
        """Initialize LLM enhancer.
        
        Args:
            agent_factory: Factory for creating LLM agents (creates default if None)
            default_provider: Default provider to use
        """
        self.agent_factory = agent_factory or AgentFactory()
        self.default_provider = default_provider
        self._prompt_cache: Dict[str, str] = {}

    def enhance(
        self,
        content: str,
        output_type: str,
        style_profile: Optional[StyleProfile] = None,
        tone: Optional[str] = None,
        length: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        url: Optional[str] = None,
    ) -> EnhancementResult:
        """Enhance content using LLM with graceful degradation.
        
        CLI flags take precedence over style profile settings.
        
        If LLM enhancement fails after all retries, falls back to
        template-only output (the original content) with a warning.
        This implements Property 13: Graceful Degradation on LLM Failure.
        
        Args:
            content: Template-rendered content to enhance
            output_type: Output type (blog, tweet, etc.)
            style_profile: Optional style profile for customization
            tone: Override tone (CLI flag precedence)
            length: Override length (CLI flag precedence)
            provider: Override provider (CLI flag precedence)
            model: Override model (CLI flag precedence)
            temperature: Override temperature
            max_tokens: Override max tokens
            url: Optional URL to include in promotional content (e.g., link to blog/linkedin)
            
        Returns:
            Enhancement result with enhanced content or original on failure
        """
        # Resolve settings with CLI flag precedence
        config = self._resolve_config(
            style_profile=style_profile,
            tone=tone,
            length=length,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Build enhancement prompt
        prompt = self.build_prompt(
            content=content,
            output_type=output_type,
            style_profile=style_profile,
            tone=config.tone,
            length=config.length,
            url=url,
        )
        
        # Get agent
        try:
            agent = self.agent_factory.create_agent(config.provider)
        except Exception as e:
            logger.error(f"Failed to create LLM agent: {e}")
            return self._create_fallback_result(
                content=content,
                error=e,
                reason="Failed to create LLM agent",
            )
        
        # Make LLM request with retry and graceful degradation
        retry_ctx = EnhancementRetryContext(
            max_attempts=3,
            base_delay=1.0,
            log_retries=True,
        )
        
        for attempt in retry_ctx:
            try:
                response = self._make_llm_call(
                    agent=agent,
                    prompt=prompt,
                    config=config,
                )
                
                # Extract enhanced content
                enhanced_content = self._extract_enhanced_content(response.content, content)
                
                return EnhancementResult(
                    content=enhanced_content,
                    success=True,
                    provider=config.provider,
                    model=response.model_used,
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    enhanced=True,
                )
                
            except PERMANENT_ERRORS as e:
                # Permanent error - don't retry, use fallback immediately
                logger.error(f"Permanent LLM error (no retry): {type(e).__name__}: {e}")
                retry_ctx.mark_fallback_used(e)
                break
                
            except TRANSIENT_ERRORS as e:
                # Transient error - check if we should retry
                if retry_ctx.should_retry(e):
                    retry_ctx.wait()
                else:
                    # All retries exhausted
                    retry_ctx.mark_fallback_used(e)
                    break
                    
            except Exception as e:
                # Unknown error - use fallback
                logger.error(f"Unknown LLM error: {type(e).__name__}: {e}")
                retry_ctx.mark_fallback_used(e)
                break
        
        # Graceful degradation: return template-only output with warning
        return self._create_fallback_result(
            content=content,
            error=retry_ctx.last_error,
            reason=retry_ctx.fallback_reason or "LLM enhancement failed",
            attempts=retry_ctx.current_attempt,
        )
    
    def _create_fallback_result(
        self,
        content: str,
        error: Optional[Exception],
        reason: str,
        attempts: int = 0,
    ) -> EnhancementResult:
        """Create a fallback result for graceful degradation.
        
        This implements Property 13: when LLM enhancement fails,
        return template-only output with a warning.
        
        Args:
            content: Original template-rendered content
            error: The error that caused fallback
            reason: Human-readable reason for fallback
            attempts: Number of retry attempts made
            
        Returns:
            EnhancementResult with original content and warning
        """
        warning_message = f"LLM enhancement failed, using template-only output. Reason: {reason}"
        if attempts > 0:
            warning_message += f" (after {attempts} attempts)"
        
        return EnhancementResult(
            content=content,
            success=False,
            enhanced=False,
            error=str(error) if error else reason,
            warnings=[warning_message],
        )
    
    def _make_llm_call(
        self,
        agent: BaseLLMAgent,
        prompt: str,
        config: EnhancementConfig,
    ) -> LLMResponse:
        """Make a single LLM call (without retry - retry is handled by caller).
        
        Args:
            agent: LLM agent to use
            prompt: Prompt to send
            config: Enhancement configuration
            
        Returns:
            LLM response
            
        Raises:
            Various LLM errors that may be transient or permanent
        """
        request = LLMRequest(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            model=config.model,
        )
        
        return agent.generate(request)
    
    def _resolve_config(
        self,
        style_profile: Optional[StyleProfile],
        tone: Optional[str],
        length: Optional[str],
        provider: Optional[str],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> EnhancementConfig:
        """Resolve configuration with CLI flag precedence.
        
        Priority order: CLI flags > style profile > defaults
        
        Args:
            style_profile: Style profile settings
            tone: CLI tone override
            length: CLI length override
            provider: CLI provider override
            model: CLI model override
            temperature: CLI temperature override
            max_tokens: CLI max_tokens override
            
        Returns:
            Resolved configuration
        """
        # Start with defaults
        resolved_provider = self.default_provider
        resolved_model = None
        resolved_temperature = DEFAULT_TEMPERATURE
        resolved_max_tokens = DEFAULT_MAX_TOKENS
        resolved_tone = DEFAULT_TONE
        resolved_length = DEFAULT_LENGTH
        
        # Apply style profile settings (if provided)
        if style_profile:
            resolved_temperature = style_profile.temperature
            resolved_max_tokens = style_profile.max_tokens
            resolved_model = style_profile.model
        
        # Apply CLI overrides (highest priority)
        if provider is not None:
            resolved_provider = provider
        if model is not None:
            resolved_model = model
        if temperature is not None:
            resolved_temperature = temperature
        if max_tokens is not None:
            resolved_max_tokens = max_tokens
        if tone is not None:
            resolved_tone = tone
        if length is not None:
            resolved_length = length
        
        return EnhancementConfig(
            provider=resolved_provider,
            model=resolved_model,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            tone=resolved_tone,
            length=resolved_length,
        )
    
    def build_prompt(
        self,
        content: str,
        output_type: str,
        style_profile: Optional[StyleProfile] = None,
        tone: str = DEFAULT_TONE,
        length: str = DEFAULT_LENGTH,
        url: Optional[str] = None,
    ) -> str:
        """Build enhancement prompt from style profile or defaults.
        
        Args:
            content: Content to enhance
            output_type: Output type (blog, tweet, etc.)
            style_profile: Optional style profile with custom prompt
            tone: Writing tone
            length: Output length preference
            url: Optional URL to include in promotional content
            
        Returns:
            Complete prompt for LLM
        """
        # Use style profile prompt template if available
        if style_profile and style_profile.prompt_template:
            return self._render_style_profile_prompt(
                style_profile=style_profile,
                content=content,
                tone=tone,
                length=length,
            )
        
        # Use default enhancement prompt
        return self._build_default_prompt(
            content=content,
            output_type=output_type,
            tone=tone,
            length=length,
            url=url,
        )
    
    def _render_style_profile_prompt(
        self,
        style_profile: StyleProfile,
        content: str,
        tone: str,
        length: str,
    ) -> str:
        """Render style profile prompt template.
        
        Args:
            style_profile: Style profile with prompt template
            content: Content to enhance
            tone: Writing tone
            length: Output length
            
        Returns:
            Rendered prompt
        """
        from jinja2 import Template
        
        template = Template(style_profile.prompt_template)
        
        return template.render(
            content=content,
            tone=tone,
            length=length,
        )
    
    def _build_default_prompt(
        self,
        content: str,
        output_type: str,
        tone: str,
        length: str,
        url: Optional[str] = None,
    ) -> str:
        """Build default enhancement prompt.
        
        Args:
            content: Content to enhance
            output_type: Output type
            tone: Writing tone
            length: Output length
            url: Optional URL to include in promotional content
            
        Returns:
            Default enhancement prompt
        """
        length_guidance = {
            "short": "Keep the content concise and to the point.",
            "medium": "Maintain a balanced length with adequate detail.",
            "long": "Provide comprehensive coverage with rich detail.",
        }
        
        tone_guidance = {
            "professional": "Use a professional, authoritative tone suitable for business contexts.",
            "casual": "Use a relaxed, conversational tone that feels approachable.",
            "technical": "Use precise technical language appropriate for expert audiences.",
            "friendly": "Use a warm, engaging tone that builds rapport with readers.",
        }
        
        # Build URL instruction only when URL is explicitly provided
        url_instruction = ""
        if url:
            url_instruction = f"""
URL REQUIREMENT:
- Include this URL in the content: {url}
- Place the URL naturally at the end or where it makes sense for a call-to-action
"""
        
        return f"""You are an expert content editor. Your task is to enhance the following {output_type} content while preserving its structure.

IMPORTANT RULES:
1. PRESERVE all structural elements (headers, lists, code blocks, timestamps, etc.)
2. IMPROVE prose quality, clarity, and engagement
3. MAINTAIN the original meaning and key information
4. DO NOT add new sections or remove existing ones
5. DO NOT change formatting markers (##, -, *, etc.)
{url_instruction}
TONE: {tone_guidance.get(tone, tone_guidance['professional'])}

LENGTH: {length_guidance.get(length, length_guidance['medium'])}

CONTENT TO ENHANCE:
{content}

ENHANCED CONTENT:"""

    def _extract_enhanced_content(
        self,
        response_content: str,
        original_content: str,
    ) -> str:
        """Extract enhanced content from LLM response.
        
        Handles cases where LLM might add extra text before/after the content.
        
        Args:
            response_content: Raw LLM response
            original_content: Original content for fallback
            
        Returns:
            Cleaned enhanced content
        """
        # Clean up response
        content = response_content.strip()
        
        # Remove common LLM prefixes
        prefixes_to_remove = [
            "Here is the enhanced content:",
            "Here's the enhanced content:",
            "Enhanced content:",
            "ENHANCED CONTENT:",
        ]
        
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
        
        # If response is empty or too short, return original
        if len(content) < len(original_content) * 0.3:
            logger.warning("LLM response too short, using original content")
            return original_content
        
        return content
    
    def estimate_cost(
        self,
        content: str,
        output_type: str,
        provider: str = "auto",
        model: Optional[str] = None,
    ) -> float:
        """Estimate cost for enhancement.
        
        Args:
            content: Content to enhance
            output_type: Output type
            provider: Provider to use
            model: Model to use
            
        Returns:
            Estimated cost in USD
        """
        try:
            agent = self.agent_factory.create_agent(provider)
            
            # Build a sample prompt to estimate tokens
            prompt = self._build_default_prompt(
                content=content,
                output_type=output_type,
                tone=DEFAULT_TONE,
                length=DEFAULT_LENGTH,
            )
            
            request = LLMRequest(
                prompt=prompt,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                model=model,
            )
            
            return agent.estimate_cost(request)
            
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")
            return 0.0
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers.
        
        Returns:
            List of available provider names
        """
        return self.agent_factory.get_available_providers()
