"""
Style profile loader for formatter module.

Loads and parses style profile Markdown files with YAML frontmatter
containing LLM settings and Jinja2 prompt templates.

Style Profile Format:
    ---
    Name: profile-name
    Temperature: 0.7
    TopP: 0.9
    MaxTokens: 2000
    Model: gpt-4
    ---
    
    # Prompt Template
    
    Your Jinja2 prompt template here with {{ variables }}.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Default profiles directory
BUILTIN_PROFILES_DIR = Path(__file__).parent / "profiles"

# Required frontmatter fields
REQUIRED_FIELDS = {"Name", "Temperature", "TopP", "MaxTokens", "Model"}

# Supported Jinja2 variables in style profiles
SUPPORTED_VARIABLES = {
    "title",
    "summary.short",
    "summary.medium", 
    "summary.long",
    "tags",
    "chapters",
    "highlights",
    "transcript",
    "content",  # Generic content placeholder
}


@dataclass
class StyleProfile:
    """Parsed style profile.
    
    Attributes:
        name: Profile name identifier
        temperature: LLM temperature setting (0.0-2.0)
        top_p: LLM top_p setting (0.0-1.0)
        max_tokens: Maximum tokens for LLM response
        model: LLM model identifier
        prompt_template: Jinja2 prompt template string
        variables: List of Jinja2 variables found in template
        source_path: Optional path to source file
    """
    
    name: str
    temperature: float
    top_p: float
    max_tokens: int
    model: str
    prompt_template: str
    variables: list[str] = field(default_factory=list)
    source_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate profile parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"TopP must be between 0.0 and 1.0, got {self.top_p}"
            )
        
        if self.max_tokens < 1:
            raise ValueError(
                f"MaxTokens must be positive, got {self.max_tokens}"
            )
        
        if not self.name:
            raise ValueError("Name cannot be empty")
        
        if not self.model:
            raise ValueError("Model cannot be empty")


class StyleProfileError(Exception):
    """Exception raised for style profile parsing errors."""
    
    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        super().__init__(f"{message}" + (f" (file: {path})" if path else ""))


class StyleProfileLoader:
    """Loads and parses style profile Markdown files.
    
    Style profiles are Markdown files with YAML frontmatter containing
    LLM settings (Name, Temperature, TopP, MaxTokens, Model) and a
    Jinja2 prompt template in the body.
    """
    
    def __init__(self, builtin_dir: Optional[Path] = None):
        """Initialize the loader.
        
        Args:
            builtin_dir: Optional custom directory for built-in profiles.
                        Defaults to pipeline/formatters/profiles/
        """
        self.builtin_dir = builtin_dir or BUILTIN_PROFILES_DIR
    
    def load(self, path: str) -> StyleProfile:
        """Load style profile from Markdown file with YAML frontmatter.
        
        Args:
            path: Path to the style profile Markdown file
            
        Returns:
            Parsed StyleProfile object
            
        Raises:
            StyleProfileError: If file cannot be read or parsed
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise StyleProfileError(f"Style profile file not found", path)
        
        if not file_path.is_file():
            raise StyleProfileError(f"Path is not a file", path)
        
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise StyleProfileError(f"Failed to read file: {e}", path)
        
        return self._parse_content(content, str(file_path))
    
    def _parse_content(self, content: str, source_path: Optional[str] = None) -> StyleProfile:
        """Parse style profile content.
        
        Args:
            content: Raw Markdown content with YAML frontmatter
            source_path: Optional source file path for error messages
            
        Returns:
            Parsed StyleProfile object
        """
        # Split frontmatter and body
        frontmatter, body = self._split_frontmatter(content, source_path)
        
        # Parse YAML frontmatter
        settings = self._parse_frontmatter(frontmatter, source_path)
        
        # Extract Jinja2 variables from template
        variables = self._extract_variables(body)
        
        # Create and return profile
        try:
            return StyleProfile(
                name=settings["Name"],
                temperature=float(settings["Temperature"]),
                top_p=float(settings["TopP"]),
                max_tokens=int(settings["MaxTokens"]),
                model=str(settings["Model"]),
                prompt_template=body.strip(),
                variables=variables,
                source_path=source_path,
            )
        except (ValueError, TypeError) as e:
            raise StyleProfileError(f"Invalid field value: {e}", source_path)
    
    def _split_frontmatter(
        self, content: str, source_path: Optional[str] = None
    ) -> tuple[str, str]:
        """Split content into frontmatter and body.
        
        Args:
            content: Raw Markdown content
            source_path: Optional source file path for error messages
            
        Returns:
            Tuple of (frontmatter_yaml, body_markdown)
        """
        # Match YAML frontmatter delimited by ---
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            raise StyleProfileError(
                "Invalid format: missing YAML frontmatter delimiters (---)",
                source_path
            )
        
        return match.group(1), match.group(2)
    
    def _parse_frontmatter(
        self, frontmatter: str, source_path: Optional[str] = None
    ) -> dict:
        """Parse YAML frontmatter.
        
        Args:
            frontmatter: YAML frontmatter string
            source_path: Optional source file path for error messages
            
        Returns:
            Dictionary of parsed settings
        """
        try:
            settings = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            raise StyleProfileError(f"Invalid YAML in frontmatter: {e}", source_path)
        
        if not isinstance(settings, dict):
            raise StyleProfileError(
                "Frontmatter must be a YAML mapping", source_path
            )
        
        # Check for required fields
        missing = REQUIRED_FIELDS - set(settings.keys())
        if missing:
            raise StyleProfileError(
                f"Missing required fields: {', '.join(sorted(missing))}",
                source_path
            )
        
        return settings
    
    def _extract_variables(self, template: str) -> list[str]:
        """Extract Jinja2 variables from template.
        
        Args:
            template: Jinja2 template string
            
        Returns:
            List of unique variable names found
        """
        # Match {{ variable }} patterns, including nested like {{ summary.short }}
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}"
        matches = re.findall(pattern, template)
        
        # Return unique variables preserving order
        seen = set()
        variables = []
        for var in matches:
            if var not in seen:
                seen.add(var)
                variables.append(var)
        
        return variables
    
    def validate(self, profile: StyleProfile) -> tuple[bool, list[str]]:
        """Validate style profile schema and template syntax.
        
        Args:
            profile: StyleProfile to validate
            
        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        errors = []
        
        # Validate temperature range
        if not 0.0 <= profile.temperature <= 2.0:
            errors.append(
                f"Temperature {profile.temperature} out of range [0.0, 2.0]"
            )
        
        # Validate top_p range
        if not 0.0 <= profile.top_p <= 1.0:
            errors.append(f"TopP {profile.top_p} out of range [0.0, 1.0]")
        
        # Validate max_tokens
        if profile.max_tokens < 1:
            errors.append(f"MaxTokens must be positive, got {profile.max_tokens}")
        
        if profile.max_tokens > 128000:
            errors.append(
                f"MaxTokens {profile.max_tokens} exceeds typical model limits"
            )
        
        # Validate name
        if not profile.name or not profile.name.strip():
            errors.append("Name cannot be empty")
        
        # Validate model
        if not profile.model or not profile.model.strip():
            errors.append("Model cannot be empty")
        
        # Validate prompt template
        if not profile.prompt_template or not profile.prompt_template.strip():
            errors.append("Prompt template cannot be empty")
        
        # Check for unsupported variables (warning, not error)
        warnings = []
        for var in profile.variables:
            # Get base variable name (before any dots)
            base_var = var.split(".")[0]
            if var not in SUPPORTED_VARIABLES and base_var not in {
                "title", "summary", "tags", "chapters", "highlights", 
                "transcript", "content"
            }:
                warnings.append(f"Unknown variable '{{{{ {var} }}}}' in template")
        
        return len(errors) == 0, errors + warnings
    
    def get_builtin(self, name: str) -> StyleProfile:
        """Get a built-in style profile by name.
        
        Args:
            name: Profile name (without .md extension)
            
        Returns:
            Parsed StyleProfile object
            
        Raises:
            StyleProfileError: If profile not found
        """
        # Normalize name
        if not name.endswith(".md"):
            name = f"{name}.md"
        
        profile_path = self.builtin_dir / name
        
        if not profile_path.exists():
            available = self.list_builtin()
            raise StyleProfileError(
                f"Built-in profile '{name}' not found. "
                f"Available profiles: {', '.join(available) if available else 'none'}"
            )
        
        return self.load(str(profile_path))
    
    def list_builtin(self) -> list[str]:
        """List available built-in style profiles.
        
        Returns:
            List of profile names (without .md extension)
        """
        if not self.builtin_dir.exists():
            return []
        
        profiles = []
        for path in self.builtin_dir.glob("*.md"):
            if path.is_file():
                profiles.append(path.stem)
        
        return sorted(profiles)
    
    def to_markdown(self, profile: StyleProfile) -> str:
        """Serialize a StyleProfile back to Markdown format.
        
        Args:
            profile: StyleProfile to serialize
            
        Returns:
            Markdown string with YAML frontmatter
        """
        frontmatter = {
            "Name": profile.name,
            "Temperature": profile.temperature,
            "TopP": profile.top_p,
            "MaxTokens": profile.max_tokens,
            "Model": profile.model,
        }
        
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        return f"---\n{yaml_str}---\n\n{profile.prompt_template}"
