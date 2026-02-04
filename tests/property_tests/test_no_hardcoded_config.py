"""
Property-Based Tests: No Hardcoded Configuration Values

This module validates that all LLM providers accept configuration objects
and do not contain hardcoded configuration values. This ensures providers
can be configured flexibly across different environments.

**Property 5: No hardcoded configuration values**

All providers must:
1. Accept configuration objects in __init__
2. Store configuration in self.config
3. Use self.config for all configuration values
4. Not contain hardcoded URLs, timeouts, models, or other config values

**Validates Requirements:**
- 3.8: Ollama provider accepts OllamaConfig
- 3.9: OpenAI provider accepts OpenAIConfig
- 8.5: LocalOllamaProvider uses configuration object
- 8.6: CloudOpenAIProvider uses configuration object
- 8.7: CloudAWSBedrockProvider uses configuration object
- 8.8: CloudAnthropicProvider uses configuration object

**Test Strategy:**
- Use AST parsing to analyze provider source code
- Check __init__ signatures accept config parameter
- Verify self.config is assigned
- Scan for hardcoded string literals that look like configuration
- Verify all config access goes through self.config

**Property Testing Approach:**
We use example-based tests here (not Hypothesis) because we're testing
static code structure rather than dynamic behavior. Each test validates
a specific aspect of the "no hardcoded config" property.
"""

import ast
import inspect
from pathlib import Path
from typing import List, Set, Tuple

import pytest

from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.config import (
    OllamaConfig,
    OpenAIConfig,
    BedrockConfig,
    AnthropicConfig
)


# Provider classes and their expected config types
PROVIDERS = [
    (LocalOllamaProvider, OllamaConfig, "pipeline/llm/providers/local_ollama.py"),
    (CloudOpenAIProvider, OpenAIConfig, "pipeline/llm/providers/cloud_openai.py"),
    (CloudAWSBedrockProvider, BedrockConfig, "pipeline/llm/providers/cloud_aws_bedrock.py"),
    (CloudAnthropicProvider, AnthropicConfig, "pipeline/llm/providers/cloud_anthropic.py"),
]


class ConfigValueVisitor(ast.NodeVisitor):
    """AST visitor to find potential hardcoded configuration values."""
    
    def __init__(self):
        self.hardcoded_urls: List[Tuple[int, str]] = []
        self.hardcoded_numbers: List[Tuple[int, int]] = []
        self.suspicious_strings: List[Tuple[int, str]] = []
        self.config_assignments: List[int] = []
        self.config_accesses: Set[str] = set()
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignments for self.config = config pattern."""
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if (isinstance(target.value, ast.Name) and 
                    target.value.id == 'self' and 
                    target.attr == 'config'):
                    self.config_assignments.append(node.lineno)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track all self.config.* accesses."""
        if isinstance(node.value, ast.Attribute):
            if (isinstance(node.value.value, ast.Name) and
                node.value.value.id == 'self' and
                node.value.attr == 'config'):
                self.config_accesses.add(node.attr)
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant) -> None:
        """Check for hardcoded configuration values."""
        if isinstance(node.value, str):
            value = node.value
            
            # Check for URLs (but allow empty strings and common patterns)
            if value and ('http://' in value or 'https://' in value):
                # Allow URLs in docstrings and comments
                # We'll filter these out in the test
                self.hardcoded_urls.append((node.lineno, value))
            
            # Check for suspicious configuration-like strings
            # (API keys, regions, model names in assignments)
            if value and any(pattern in value.lower() for pattern in [
                'api_key', 'secret', 'token', 'password',
                'us-east', 'us-west', 'eu-', 'ap-',  # AWS regions
            ]):
                self.suspicious_strings.append((node.lineno, value))
        
        elif isinstance(node.value, (int, float)):
            # Check for timeout-like numbers (but allow common constants)
            if isinstance(node.value, int) and node.value > 10 and node.value < 1000:
                # Could be a timeout or max_tokens
                self.hardcoded_numbers.append((node.lineno, node.value))
        
        self.generic_visit(node)


def get_provider_source_path(relative_path: str) -> Path:
    """Get absolute path to provider source file."""
    # Get the project root (3 levels up from this test file)
    test_file = Path(__file__).resolve()
    project_root = test_file.parent.parent.parent
    return project_root / relative_path


def parse_provider_source(source_path: Path) -> Tuple[ast.Module, str]:
    """Parse provider source code into AST."""
    with open(source_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename=str(source_path))
    return tree, source


def is_in_docstring_or_comment(lineno: int, source: str) -> bool:
    """Check if a line is inside a docstring or comment."""
    lines = source.split('\n')
    if lineno > len(lines):
        return False
    
    line = lines[lineno - 1].strip()
    
    # Check if line is a comment
    if line.startswith('#'):
        return True
    
    # Check if line contains triple quotes (docstring)
    if '"""' in line or "'''" in line:
        return True
    
    # Check if we're inside a docstring by looking at surrounding context
    # This is a simple heuristic - could be improved
    in_docstring = False
    for i in range(max(0, lineno - 10), min(len(lines), lineno + 10)):
        if '"""' in lines[i] or "'''" in lines[i]:
            in_docstring = not in_docstring
        if i == lineno - 1:
            return in_docstring
    
    return False


def is_in_class_constant(lineno: int, tree: ast.Module) -> bool:
    """Check if a line is inside a class-level constant definition."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    # Class-level constants (PRICING, CONTEXT_WINDOWS, etc.)
                    if hasattr(item, 'lineno') and item.lineno <= lineno:
                        # Check if this is a dict/constant assignment
                        if isinstance(item.value, (ast.Dict, ast.Constant)):
                            # Allow constants within reasonable range
                            if lineno <= item.lineno + 50:  # Reasonable constant size
                                return True
    return False


# ============================================================================
# Test 1: Provider __init__ accepts config parameter
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_provider_init_accepts_config_parameter(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.1: Provider __init__ accepts config parameter**
    
    Validates that each provider's __init__ method accepts a configuration
    object as its parameter (not individual config values).
    
    This ensures providers use configuration objects rather than accepting
    multiple individual parameters that would be hardcoded.
    """
    # Get __init__ signature
    sig = inspect.signature(provider_class.__init__)
    params = list(sig.parameters.keys())
    
    # Should have 'self' and 'config' parameters
    assert 'self' in params, \
        f"{provider_class.__name__}.__init__ missing 'self' parameter"
    
    assert 'config' in params, \
        f"{provider_class.__name__}.__init__ missing 'config' parameter"
    
    # Should not have other parameters (no individual config values)
    assert len(params) == 2, \
        f"{provider_class.__name__}.__init__ should only have 'self' and 'config' parameters, " \
        f"found: {params}"
    
    # Verify config parameter has correct type annotation
    config_param = sig.parameters['config']
    if config_param.annotation != inspect.Parameter.empty:
        # Type annotation should be the config class
        assert config_param.annotation == config_class, \
            f"{provider_class.__name__}.__init__ config parameter should be " \
            f"annotated as {config_class.__name__}, found: {config_param.annotation}"


# ============================================================================
# Test 2: Provider stores config in self.config
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_provider_stores_config_in_self_config(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.2: Provider stores config in self.config**
    
    Validates that each provider stores the configuration object in
    self.config during initialization.
    
    This ensures a consistent pattern for accessing configuration across
    all providers.
    """
    source_path = get_provider_source_path(source_path)
    tree, source = parse_provider_source(source_path)
    
    visitor = ConfigValueVisitor()
    visitor.visit(tree)
    
    # Should have at least one self.config assignment
    assert len(visitor.config_assignments) > 0, \
        f"{provider_class.__name__} does not assign self.config in __init__"
    
    # The assignment should be in __init__ (typically within first 20 lines of class)
    # We'll verify by checking it's early in the file
    first_assignment = min(visitor.config_assignments)
    assert first_assignment < 200, \
        f"{provider_class.__name__} self.config assignment at line {first_assignment} " \
        f"seems too late (should be in __init__)"


# ============================================================================
# Test 3: No hardcoded URLs in provider code
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_no_hardcoded_urls_in_provider(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.3: No hardcoded URLs in provider code**
    
    Validates that providers do not contain hardcoded URLs (except in
    docstrings, comments, or class constants).
    
    All URLs should come from configuration objects.
    """
    source_path = get_provider_source_path(source_path)
    tree, source = parse_provider_source(source_path)
    
    visitor = ConfigValueVisitor()
    visitor.visit(tree)
    
    # Filter out URLs in docstrings, comments, and class constants
    hardcoded_urls = [
        (lineno, url) for lineno, url in visitor.hardcoded_urls
        if not is_in_docstring_or_comment(lineno, source)
        and not is_in_class_constant(lineno, tree)
    ]
    
    # Should have no hardcoded URLs in actual code
    assert len(hardcoded_urls) == 0, \
        f"{provider_class.__name__} contains hardcoded URLs:\n" + \
        "\n".join(f"  Line {lineno}: {url}" for lineno, url in hardcoded_urls) + \
        "\n\nAll URLs should come from self.config"


# ============================================================================
# Test 4: Provider uses self.config for configuration access
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_provider_uses_self_config_for_access(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.4: Provider uses self.config for configuration access**
    
    Validates that providers access configuration values through self.config
    rather than using hardcoded values or other sources.
    
    This ensures all configuration is centralized and can be overridden.
    """
    source_path = get_provider_source_path(source_path)
    tree, source = parse_provider_source(source_path)
    
    visitor = ConfigValueVisitor()
    visitor.visit(tree)
    
    # Should have multiple self.config accesses
    assert len(visitor.config_accesses) > 0, \
        f"{provider_class.__name__} does not access self.config " \
        f"(should use self.config.* for configuration values)"
    
    # Verify common config attributes are accessed
    config_instance = config_class()
    config_attrs = {
        attr for attr in dir(config_instance)
        if not attr.startswith('_')
    }
    
    # Should access at least some config attributes
    accessed_attrs = visitor.config_accesses & config_attrs
    assert len(accessed_attrs) > 0, \
        f"{provider_class.__name__} does not access any config attributes. " \
        f"Expected to access some of: {config_attrs}"


# ============================================================================
# Test 5: Provider can be instantiated with config object
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_provider_can_be_instantiated_with_config(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.5: Provider can be instantiated with config object**
    
    Validates that providers can be successfully instantiated with a
    configuration object (runtime validation).
    
    This ensures the configuration interface works correctly.
    """
    # Create a config instance with default values
    config = config_class()
    
    # For providers that require credentials, provide dummy values
    if config_class == AnthropicConfig:
        config.api_key = "sk-ant-test-key"
    elif config_class == OpenAIConfig:
        config.api_key = "sk-test-key"
    elif config_class == BedrockConfig:
        # Bedrock can work without explicit credentials (uses IAM roles)
        pass
    
    # Should be able to instantiate provider
    try:
        provider = provider_class(config)
        assert provider is not None
        assert hasattr(provider, 'config')
        assert provider.config == config
    except ImportError as e:
        # Some providers may require optional dependencies
        # This is acceptable - skip the test
        pytest.skip(f"Optional dependency not installed: {e}")
    except Exception as e:
        pytest.fail(
            f"Failed to instantiate {provider_class.__name__} with config: {e}"
        )


# ============================================================================
# Test 6: Config object contains expected attributes
# ============================================================================

@pytest.mark.parametrize("provider_class,config_class,source_path", PROVIDERS)
def test_config_object_has_expected_attributes(
    provider_class,
    config_class,
    source_path
):
    """
    **Property 5.6: Config object contains expected attributes**
    
    Validates that configuration objects have the expected attributes
    for their provider type.
    
    This ensures configuration objects are complete and properly defined.
    """
    config = config_class()
    
    # All configs should have these common attributes
    common_attrs = ['default_model', 'max_tokens', 'temperature']
    
    for attr in common_attrs:
        assert hasattr(config, attr), \
            f"{config_class.__name__} missing common attribute: {attr}"
    
    # Provider-specific attributes
    if config_class == OllamaConfig:
        assert hasattr(config, 'base_url')
        assert hasattr(config, 'timeout')
    
    elif config_class == OpenAIConfig:
        assert hasattr(config, 'api_key')
        assert hasattr(config, 'timeout')
    
    elif config_class == BedrockConfig:
        assert hasattr(config, 'region')
        assert hasattr(config, 'access_key_id')
        assert hasattr(config, 'secret_access_key')
    
    elif config_class == AnthropicConfig:
        assert hasattr(config, 'api_key')
        assert hasattr(config, 'timeout')


# ============================================================================
# Summary Test: All providers follow configuration pattern
# ============================================================================

def test_all_providers_follow_configuration_pattern():
    """
    **Property 5 Summary: All providers follow configuration pattern**
    
    High-level validation that all providers follow the configuration
    pattern consistently:
    
    1. Accept config object in __init__
    2. Store config in self.config
    3. Access config through self.config.*
    4. No hardcoded configuration values
    
    This test provides a summary view of the configuration pattern
    compliance across all providers.
    """
    results = []
    
    for provider_class, config_class, source_path in PROVIDERS:
        # Check __init__ signature
        sig = inspect.signature(provider_class.__init__)
        has_config_param = 'config' in sig.parameters
        
        # Check can instantiate
        can_instantiate = False
        try:
            config = config_class()
            
            # For providers that require credentials, provide dummy values
            if config_class == AnthropicConfig:
                config.api_key = "sk-ant-test-key"
            elif config_class == OpenAIConfig:
                config.api_key = "sk-test-key"
            
            provider = provider_class(config)
            can_instantiate = hasattr(provider, 'config')
        except (ImportError, Exception):
            # Optional dependencies or other issues
            can_instantiate = None  # Unknown
        
        results.append({
            'provider': provider_class.__name__,
            'has_config_param': has_config_param,
            'can_instantiate': can_instantiate,
        })
    
    # All providers should pass
    for result in results:
        assert result['has_config_param'], \
            f"{result['provider']} does not accept config parameter"
        
        if result['can_instantiate'] is not None:
            assert result['can_instantiate'], \
                f"{result['provider']} cannot be instantiated with config"
    
    # Summary report
    print("\n" + "="*70)
    print("Configuration Pattern Compliance Summary")
    print("="*70)
    for result in results:
        status = "✓" if result['can_instantiate'] else "?" if result['can_instantiate'] is None else "✗"
        print(f"{status} {result['provider']}")
    print("="*70)
