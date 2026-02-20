"""
Template engine for formatter module.

Manages Jinja2 template loading, rendering, and validation for
generating platform-specific content from enriched data.

The engine supports:
- Loading templates from built-in and custom directories
- Custom Jinja2 filters for content formatting
- Template validation before rendering
- Fallback to default templates when custom not found
"""

import re
from pathlib import Path
from typing import Any, Optional

from jinja2 import (
    Environment,
    FileSystemLoader,
    TemplateNotFound,
    TemplateSyntaxError,
    select_autoescape,
    Undefined,
)


# Default templates directory
BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Output types that have templates
TEMPLATE_OUTPUT_TYPES = [
    "blog",
    "tweet",
    "youtube",
    "seo",
    "linkedin",
    "newsletter",
    "chapters",
    "transcript_clean",
    "podcast_notes",
    "meeting_minutes",
    "slides",
    "notion",
    "obsidian",
    "quote_cards",
    "video_script",
    "tiktok_script",
    "ai_video_script",
]


class TemplateEngineError(Exception):
    """Exception raised for template engine errors."""

    def __init__(self, message: str, template_name: Optional[str] = None):
        self.template_name = template_name
        super().__init__(
            f"{message}" + (f" (template: {template_name})" if template_name else "")
        )


# ============================================================================
# Custom Jinja2 Filters
# ============================================================================


def truncate_words(text: str, max_words: int, suffix: str = "...") -> str:
    """Truncate text to a maximum number of words.
    
    Args:
        text: Text to truncate
        max_words: Maximum number of words
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    return " ".join(words[:max_words]) + suffix


def truncate_chars(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to a maximum number of characters at word boundary.
    
    Args:
        text: Text to truncate
        max_chars: Maximum number of characters
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_chars:
        return text
    
    # Find last space before max_chars
    truncated = text[: max_chars - len(suffix)]
    last_space = truncated.rfind(" ")
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    if not isinstance(seconds, (int, float)):
        return "00:00"
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def hashtag(text: str) -> str:
    """Convert text to a hashtag format.
    
    Args:
        text: Text to convert
        
    Returns:
        Hashtag string (e.g., "#MachineLearning")
    """
    if not text:
        return ""
    
    # Remove special characters and convert to PascalCase
    words = re.sub(r"[^\w\s]", "", text).split()
    pascal_case = "".join(word.capitalize() for word in words)
    
    return f"#{pascal_case}"


def bullet_list(items: list, prefix: str = "• ") -> str:
    """Format a list as bullet points.
    
    Args:
        items: List of items
        prefix: Bullet prefix character
        
    Returns:
        Formatted bullet list string
    """
    if not items:
        return ""
    
    return "\n".join(f"{prefix}{item}" for item in items)


def numbered_list(items: list, start: int = 1) -> str:
    """Format a list as numbered items.
    
    Args:
        items: List of items
        start: Starting number
        
    Returns:
        Formatted numbered list string
    """
    if not items:
        return ""
    
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=start))


def strip_html(text: str) -> str:
    """Remove HTML tags from text.
    
    Args:
        text: Text with potential HTML
        
    Returns:
        Text with HTML tags removed
    """
    if not text:
        return ""
    
    return re.sub(r"<[^>]+>", "", text)


def sentence_case(text: str) -> str:
    """Convert text to sentence case.
    
    Args:
        text: Text to convert
        
    Returns:
        Sentence case text
    """
    if not text:
        return ""
    
    return text[0].upper() + text[1:].lower() if len(text) > 1 else text.upper()


def title_case(text: str) -> str:
    """Convert text to title case.
    
    Args:
        text: Text to convert
        
    Returns:
        Title case text
    """
    if not text:
        return ""
    
    return text.title()


def join_with(items: list, separator: str = ", ", last_separator: str = " and ") -> str:
    """Join list items with separators, using different separator for last item.
    
    Args:
        items: List of items
        separator: Separator between items
        last_separator: Separator before last item
        
    Returns:
        Joined string
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return str(items[0])
    
    if len(items) == 2:
        return f"{items[0]}{last_separator}{items[1]}"
    
    return separator.join(str(item) for item in items[:-1]) + last_separator + str(items[-1])


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        First sentence
    """
    if not text:
        return ""
    
    # Match first sentence ending with . ! or ?
    match = re.match(r"^[^.!?]*[.!?]", text)
    if match:
        return match.group(0).strip()
    
    return text.strip()


def word_count(text: str) -> int:
    """Count words in text.
    
    Args:
        text: Text to count
        
    Returns:
        Word count
    """
    if not text:
        return 0
    
    return len(text.split())


def reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes.
    
    Args:
        text: Text to estimate
        words_per_minute: Reading speed
        
    Returns:
        Estimated minutes
    """
    if not text:
        return 0
    
    words = word_count(text)
    minutes = max(1, round(words / words_per_minute))
    return minutes


# ============================================================================
# Template Engine
# ============================================================================


class TemplateEngine:
    """Manages Jinja2 template loading and rendering.
    
    The engine loads templates from a built-in directory and optionally
    from a custom directory. Custom templates take precedence over built-in.
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        custom_templates_dir: Optional[Path] = None,
    ):
        """Initialize the template engine.
        
        Args:
            templates_dir: Path to built-in templates directory.
                          Defaults to pipeline/formatters/templates/
            custom_templates_dir: Optional path to custom templates directory.
                                 Custom templates override built-in.
        """
        self.templates_dir = templates_dir or BUILTIN_TEMPLATES_DIR
        self.custom_templates_dir = custom_templates_dir
        
        # Build loader paths (custom first for precedence)
        loader_paths = []
        if custom_templates_dir and custom_templates_dir.exists():
            loader_paths.append(str(custom_templates_dir))
        if self.templates_dir.exists():
            loader_paths.append(str(self.templates_dir))
        
        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(loader_paths) if loader_paths else None,
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # Register custom filters
        self._register_filters()
    
    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self.env.filters["truncate_words"] = truncate_words
        self.env.filters["truncate_chars"] = truncate_chars
        self.env.filters["format_timestamp"] = format_timestamp
        self.env.filters["hashtag"] = hashtag
        self.env.filters["bullet_list"] = bullet_list
        self.env.filters["numbered_list"] = numbered_list
        self.env.filters["strip_html"] = strip_html
        self.env.filters["sentence_case"] = sentence_case
        self.env.filters["title_case"] = title_case
        self.env.filters["join_with"] = join_with
        self.env.filters["first_sentence"] = extract_first_sentence
        self.env.filters["word_count"] = word_count
        self.env.filters["reading_time"] = reading_time
    
    def render(self, output_type: str, context: dict) -> str:
        """Render template with enriched content context.
        
        Args:
            output_type: Output type (e.g., "blog", "tweet")
            context: Dictionary with enriched content data
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateEngineError: If template not found or rendering fails
        """
        template = self.get_template(output_type)
        
        try:
            return template.render(**context)
        except Exception as e:
            raise TemplateEngineError(
                f"Failed to render template: {e}",
                template_name=f"{output_type}.j2"
            )
    
    def get_template(self, output_type: str):
        """Load template for output type.
        
        Args:
            output_type: Output type (e.g., "blog", "tweet")
            
        Returns:
            Jinja2 Template object
            
        Raises:
            TemplateEngineError: If template not found
        """
        # Normalize output type (convert dashes to underscores for filename)
        template_name = output_type.replace("-", "_") + ".j2"
        
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            raise TemplateEngineError(
                f"Template not found for output type '{output_type}'",
                template_name=template_name
            )
    
    def validate_template(self, template_path: str) -> tuple[bool, list[str]]:
        """Validate template syntax.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        path = Path(template_path)
        
        if not path.exists():
            errors.append(f"Template file not found: {template_path}")
            return False, errors
        
        if not path.is_file():
            errors.append(f"Path is not a file: {template_path}")
            return False, errors
        
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"Failed to read template: {e}")
            return False, errors
        
        try:
            self.env.parse(content)
        except TemplateSyntaxError as e:
            errors.append(f"Template syntax error at line {e.lineno}: {e.message}")
            return False, errors
        
        return True, []
    
    def template_exists(self, output_type: str) -> bool:
        """Check if template exists for output type.
        
        Args:
            output_type: Output type (e.g., "blog", "tweet")
            
        Returns:
            True if template exists
        """
        template_name = output_type.replace("-", "_") + ".j2"
        
        try:
            self.env.get_template(template_name)
            return True
        except TemplateNotFound:
            return False
    
    def list_templates(self) -> list[str]:
        """List available templates.
        
        Returns:
            List of output types with available templates
        """
        templates = []
        
        for output_type in TEMPLATE_OUTPUT_TYPES:
            if self.template_exists(output_type.replace("_", "-")):
                templates.append(output_type.replace("_", "-"))
        
        return templates
    
    def get_template_variables(self, output_type: str) -> list[str]:
        """Get list of variables used in a template.
        
        Args:
            output_type: Output type (e.g., "blog", "tweet")
            
        Returns:
            List of variable names used in template
        """
        template = self.get_template(output_type)
        
        # Get undeclared variables from template AST
        from jinja2 import meta
        
        ast = self.env.parse(template.source)
        variables = meta.find_undeclared_variables(ast)
        
        return sorted(list(variables))
    
    def render_string(self, template_string: str, context: dict) -> str:
        """Render a template string directly.
        
        Args:
            template_string: Jinja2 template string
            context: Dictionary with context data
            
        Returns:
            Rendered string
            
        Raises:
            TemplateEngineError: If rendering fails
        """
        try:
            template = self.env.from_string(template_string)
            return template.render(**context)
        except Exception as e:
            raise TemplateEngineError(f"Failed to render template string: {e}")
