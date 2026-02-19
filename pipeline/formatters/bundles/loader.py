"""
Bundle configuration loader.

Loads bundle definitions from YAML files and provides validation
for bundle configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Valid output types that can be included in bundles
VALID_OUTPUT_TYPES = [
    "blog",
    "tweet",
    "youtube",
    "seo",
    "linkedin",
    "newsletter",
    "chapters",
    "transcript-clean",
    "podcast-notes",
    "meeting-minutes",
    "slides",
    "notion",
    "obsidian",
    "quote-cards",
    "video-script",
    "tiktok-script",
]


class BundleNotFoundError(Exception):
    """Raised when a requested bundle name doesn't exist."""

    def __init__(self, bundle_name: str, available_bundles: list[str]):
        self.bundle_name = bundle_name
        self.available_bundles = available_bundles
        available_str = ", ".join(sorted(available_bundles)) if available_bundles else "none"
        super().__init__(
            f"Bundle '{bundle_name}' not found. "
            f"Available bundles: {available_str}"
        )


@dataclass
class BundleConfig:
    """Configuration for a named bundle.
    
    Attributes:
        name: Unique bundle identifier (e.g., "blog-launch")
        description: Human-readable description of the bundle
        outputs: List of output types to generate
    """
    
    name: str
    description: str
    outputs: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate outputs list is not empty."""
        if not self.outputs:
            raise ValueError(f"Bundle '{self.name}' must have at least one output type")


class BundleLoader:
    """Loads bundle configurations from YAML files.
    
    Bundle configurations are loaded from:
    1. Default bundles (built-in, shipped with the package)
    2. User bundles (.content-pipeline/bundles.yaml in workspace)
    3. Custom bundles (specified via --bundles-config flag)
    
    User and custom bundles override defaults with the same name.
    """
    
    # Default paths for bundle configuration files
    DEFAULT_BUNDLES_PATH = Path(__file__).parent / "default.yaml"
    USER_BUNDLES_PATH = Path(".content-pipeline/bundles.yaml")
    
    def __init__(self, custom_bundles_path: Optional[str] = None):
        """Initialize the bundle loader.
        
        Args:
            custom_bundles_path: Optional path to custom bundles YAML file
        """
        self._custom_bundles_path = Path(custom_bundles_path) if custom_bundles_path else None
        self._bundles: dict[str, BundleConfig] = {}
        self._load_all_bundles()
    
    def _load_all_bundles(self) -> None:
        """Load bundles from all sources in priority order."""
        # Load default bundles first
        self._load_bundles_from_file(self.DEFAULT_BUNDLES_PATH)
        
        # Load user bundles (override defaults)
        if self.USER_BUNDLES_PATH.exists():
            self._load_bundles_from_file(self.USER_BUNDLES_PATH)
        
        # Load custom bundles (highest priority)
        if self._custom_bundles_path and self._custom_bundles_path.exists():
            self._load_bundles_from_file(self._custom_bundles_path)
    
    def _load_bundles_from_file(self, path: Path) -> None:
        """Load bundles from a YAML file.
        
        Args:
            path: Path to the YAML file
        """
        if not path.exists():
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in bundle file {path}: {e}")
        
        if not data or "bundles" not in data:
            return
        
        bundles_data = data["bundles"]
        if not isinstance(bundles_data, dict):
            raise ValueError(f"'bundles' must be a dictionary in {path}")
        
        for name, config in bundles_data.items():
            if not isinstance(config, dict):
                raise ValueError(f"Bundle '{name}' must be a dictionary in {path}")
            
            bundle = BundleConfig(
                name=name,
                description=config.get("description", ""),
                outputs=config.get("outputs", []),
            )
            
            # Validate the bundle
            is_valid, errors = self.validate_bundle(bundle)
            if not is_valid:
                raise ValueError(f"Invalid bundle '{name}' in {path}: {'; '.join(errors)}")
            
            self._bundles[name] = bundle

    
    def load_bundle(self, name: str) -> BundleConfig:
        """Load a bundle by name.
        
        Args:
            name: Bundle name (e.g., "blog-launch")
            
        Returns:
            BundleConfig with outputs list
            
        Raises:
            BundleNotFoundError: If bundle name doesn't exist
        """
        if name not in self._bundles:
            raise BundleNotFoundError(name, list(self._bundles.keys()))
        
        return self._bundles[name]
    
    def list_bundles(self) -> list[BundleConfig]:
        """List all available bundles.
        
        Returns:
            List of all bundle configurations, sorted by name
        """
        return sorted(self._bundles.values(), key=lambda b: b.name)
    
    def get_bundle_names(self) -> list[str]:
        """Get list of available bundle names.
        
        Returns:
            Sorted list of bundle names
        """
        return sorted(self._bundles.keys())
    
    def has_bundle(self, name: str) -> bool:
        """Check if a bundle exists.
        
        Args:
            name: Bundle name to check
            
        Returns:
            True if bundle exists, False otherwise
        """
        return name in self._bundles
    
    @staticmethod
    def validate_bundle(bundle: BundleConfig) -> tuple[bool, list[str]]:
        """Validate that all output types in bundle are valid.
        
        Args:
            bundle: Bundle configuration to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        
        if not bundle.outputs:
            errors.append("Bundle must have at least one output type")
            return False, errors
        
        invalid_types = [
            output for output in bundle.outputs
            if output not in VALID_OUTPUT_TYPES
        ]
        
        if invalid_types:
            errors.append(
                f"Invalid output types: {', '.join(invalid_types)}. "
                f"Valid types: {', '.join(VALID_OUTPUT_TYPES)}"
            )
        
        # Check for duplicates
        seen = set()
        duplicates = []
        for output in bundle.outputs:
            if output in seen:
                duplicates.append(output)
            seen.add(output)
        
        if duplicates:
            errors.append(f"Duplicate output types: {', '.join(duplicates)}")
        
        return len(errors) == 0, errors
    
    def format_bundle_list(self) -> str:
        """Format bundle list for CLI display.
        
        Returns:
            Formatted string showing all bundles and their outputs
        """
        lines = ["Available bundles:", ""]
        
        for bundle in self.list_bundles():
            lines.append(f"  {bundle.name}")
            if bundle.description:
                lines.append(f"    {bundle.description}")
            lines.append(f"    Outputs: {', '.join(bundle.outputs)}")
            lines.append("")
        
        return "\n".join(lines)
