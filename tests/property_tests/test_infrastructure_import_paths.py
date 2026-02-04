"""
Property-Based Tests for Infrastructure Import Paths

This module validates that all code uses correct import paths after
the infrastructure refactoring. It ensures:
- No imports from old paths (pipeline.enrichment.agents)
- All LLM imports use pipeline.llm
- All transcription imports use pipeline.transcription (when implemented)
- No cross-domain dependencies

**Validates: Requirements 1.8, 1.9, 2.7, 2.8, 5.1-5.8**
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Set
import pytest


# Old paths that should no longer be used
DEPRECATED_IMPORT_PATHS = {
    "pipeline.enrichment.agents",
    "pipeline.enrichment.agents.base",
    "pipeline.enrichment.agents.factory",
    "pipeline.enrichment.agents.local_ollama_agent",
    "pipeline.enrichment.agents.cloud_openai_agent",
    "pipeline.enrichment.agents.cloud_anthropic_agent",
    "pipeline.enrichment.agents.cloud_aws_bedrock_agent",
    "pipeline.transcribers.adapters",  # Will be deprecated when transcription refactored
    "pipeline.transcribers.adapters.base",
    "pipeline.transcribers.adapters.local_whisper",
    "pipeline.transcribers.adapters.openai_whisper",
    "pipeline.transcribers.adapters.aws_transcribe",
}

# New paths that should be used
CORRECT_LLM_PATHS = {
    "pipeline.llm",
    "pipeline.llm.config",
    "pipeline.llm.factory",
    "pipeline.llm.errors",
    "pipeline.llm.providers",
    "pipeline.llm.providers.base",
    "pipeline.llm.providers.local_ollama",
    "pipeline.llm.providers.cloud_openai",
    "pipeline.llm.providers.cloud_anthropic",
    "pipeline.llm.providers.cloud_aws_bedrock",
}

# Directories to check
DIRECTORIES_TO_CHECK = [
    "pipeline/enrichment",
    "pipeline/formatters",
    "pipeline/transcribers",
    "pipeline/extractors",
    "cli",
    "tests/pipeline/enrichment",
    "tests/pipeline/formatters",
    "tests/cli",
]


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""
    
    def __init__(self):
        self.imports: List[Tuple[str, int]] = []  # (module_path, line_number)
    
    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statement."""
        if node.module:
            self.imports.append((node.module, node.lineno))
        self.generic_visit(node)


def get_python_files(directory: str) -> List[Path]:
    """Get all Python files in directory recursively.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    root = Path(directory)
    if not root.exists():
        return []
    
    python_files = []
    for path in root.rglob("*.py"):
        # Skip __pycache__ and other generated files
        if "__pycache__" in str(path):
            continue
        python_files.append(path)
    
    return python_files


def extract_imports(file_path: Path) -> List[Tuple[str, int]]:
    """Extract all import statements from a Python file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of (module_path, line_number) tuples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except SyntaxError:
        # Skip files with syntax errors
        return []


def find_deprecated_imports() -> List[Tuple[Path, str, int]]:
    """Find all uses of deprecated import paths.
    
    Returns:
        List of (file_path, import_path, line_number) tuples
    """
    violations = []
    
    for directory in DIRECTORIES_TO_CHECK:
        python_files = get_python_files(directory)
        
        for file_path in python_files:
            imports = extract_imports(file_path)
            
            for import_path, line_num in imports:
                # Check if import uses deprecated path
                for deprecated_path in DEPRECATED_IMPORT_PATHS:
                    if import_path == deprecated_path or import_path.startswith(deprecated_path + "."):
                        violations.append((file_path, import_path, line_num))
    
    return violations


def find_llm_imports_outside_infrastructure() -> List[Tuple[Path, str, int]]:
    """Find LLM-related imports that don't use pipeline.llm.
    
    Returns:
        List of (file_path, import_path, line_number) tuples
    """
    violations = []
    
    # Keywords that indicate LLM-related imports
    llm_keywords = ["agent", "llm", "anthropic", "claude", "bedrock", "ollama"]
    
    # Keywords that indicate transcription-related imports (not LLM)
    transcription_keywords = ["whisper", "transcribe", "transcription"]
    
    for directory in DIRECTORIES_TO_CHECK:
        python_files = get_python_files(directory)
        
        for file_path in python_files:
            imports = extract_imports(file_path)
            
            for import_path, line_num in imports:
                # Skip if already using correct path
                if import_path.startswith("pipeline.llm"):
                    continue
                
                # Skip if it's a transcription-related import (not LLM)
                if any(keyword in import_path.lower() for keyword in transcription_keywords):
                    continue
                
                # Check if import looks LLM-related but doesn't use pipeline.llm
                if any(keyword in import_path.lower() for keyword in llm_keywords):
                    if import_path.startswith("pipeline."):
                        violations.append((file_path, import_path, line_num))
    
    return violations


def check_cross_domain_dependencies() -> List[Tuple[Path, str, str, int]]:
    """Check for inappropriate cross-domain dependencies.
    
    Returns:
        List of (file_path, from_domain, to_domain, line_number) tuples
    """
    violations = []
    
    # Define domain boundaries
    domains = {
        "enrichment": "pipeline/enrichment",
        "formatters": "pipeline/formatters",
        "transcribers": "pipeline/transcribers",
        "extractors": "pipeline/extractors",
        "llm": "pipeline/llm",
        "transcription": "pipeline/transcription",
    }
    
    # Allowed dependencies (domain -> allowed imports)
    # Note: Each domain can import from itself (internal imports)
    allowed_dependencies = {
        "enrichment": {"enrichment", "llm", "config", "utils", "output"},
        "formatters": {"formatters", "llm", "config", "utils", "output"},
        "transcribers": {"transcribers", "transcription", "config", "utils", "output"},
        "extractors": {"extractors", "transcription", "config", "utils", "output"},
        "llm": {"llm", "config", "utils"},  # LLM should not depend on enrichment
        "transcription": {"transcription", "config", "utils"},  # Transcription should not depend on transcribers
    }
    
    for domain_name, domain_path in domains.items():
        python_files = get_python_files(domain_path)
        
        for file_path in python_files:
            imports = extract_imports(file_path)
            
            for import_path, line_num in imports:
                # Check if importing from another domain
                if import_path.startswith("pipeline."):
                    parts = import_path.split(".")
                    if len(parts) >= 2:
                        target_domain = parts[1]
                        
                        # Check if this is a disallowed cross-domain import
                        if target_domain in domains:
                            if target_domain not in allowed_dependencies.get(domain_name, set()):
                                violations.append((
                                    file_path,
                                    domain_name,
                                    target_domain,
                                    line_num
                                ))
    
    return violations


# ============================================================================
# Property Tests
# ============================================================================

def test_no_deprecated_import_paths():
    """
    **Property 4a: No deprecated import paths are used**
    
    Validates that no code uses old import paths like:
    - pipeline.enrichment.agents
    - pipeline.transcribers.adapters
    
    **Validates: Requirements 1.8, 1.9, 2.7, 2.8, 5.1-5.7**
    """
    violations = find_deprecated_imports()
    
    if violations:
        error_msg = "Found deprecated import paths:\n"
        for file_path, import_path, line_num in violations:
            error_msg += f"  {file_path}:{line_num} - {import_path}\n"
        error_msg += "\nThese imports should be updated to use new infrastructure paths."
        pytest.fail(error_msg)


def test_llm_imports_use_correct_path():
    """
    **Property 4b: All LLM imports use pipeline.llm**
    
    Validates that all LLM-related imports use the correct
    pipeline.llm path instead of pipeline.enrichment.agents.
    
    **Validates: Requirements 1.8, 5.1, 5.2, 5.5, 5.7**
    """
    violations = find_llm_imports_outside_infrastructure()
    
    if violations:
        error_msg = "Found LLM-related imports not using pipeline.llm:\n"
        for file_path, import_path, line_num in violations:
            error_msg += f"  {file_path}:{line_num} - {import_path}\n"
        error_msg += "\nThese imports should use pipeline.llm instead."
        pytest.fail(error_msg)


def test_no_cross_domain_dependencies():
    """
    **Property 4c: No inappropriate cross-domain dependencies**
    
    Validates that domains don't import from each other inappropriately:
    - LLM infrastructure should not import from enrichment
    - Transcription infrastructure should not import from transcribers
    - Enrichment should use LLM infrastructure, not vice versa
    
    **Validates: Requirements 1.10, 1.11, 5.8**
    """
    violations = check_cross_domain_dependencies()
    
    if violations:
        error_msg = "Found inappropriate cross-domain dependencies:\n"
        for file_path, from_domain, to_domain, line_num in violations:
            error_msg += f"  {file_path}:{line_num} - {from_domain} -> {to_domain}\n"
        error_msg += "\nThese dependencies violate domain boundaries."
        pytest.fail(error_msg)


def test_infrastructure_exports_are_complete():
    """
    **Property 4d: Infrastructure __init__.py exports are complete**
    
    Validates that pipeline.llm.__init__.py exports all necessary
    classes and functions for external use.
    
    **Validates: Requirements 5.8**
    """
    llm_init_path = Path("pipeline/llm/__init__.py")
    
    if not llm_init_path.exists():
        pytest.fail("pipeline/llm/__init__.py does not exist")
    
    with open(llm_init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Required exports
    required_exports = [
        "BaseLLMProvider",
        "LLMRequest",
        "LLMResponse",
        "LLMProviderFactory",
        "LLMConfig",
        "LLMError",
        "ConfigurationError",
        "ProviderError",
    ]
    
    missing_exports = []
    for export in required_exports:
        if export not in content:
            missing_exports.append(export)
    
    if missing_exports:
        error_msg = f"Missing exports in pipeline/llm/__init__.py:\n"
        for export in missing_exports:
            error_msg += f"  - {export}\n"
        pytest.fail(error_msg)


def test_all_providers_use_config_objects():
    """
    **Property 4e: All providers accept config objects**
    
    Validates that all provider __init__ methods accept
    configuration objects instead of individual parameters.
    
    **Validates: Requirements 3.1-3.5, 8.1-8.8**
    """
    provider_files = [
        "pipeline/llm/providers/local_ollama.py",
        "pipeline/llm/providers/cloud_openai.py",
        "pipeline/llm/providers/cloud_anthropic.py",
        "pipeline/llm/providers/cloud_aws_bedrock.py",
    ]
    
    violations = []
    
    for provider_file in provider_files:
        path = Path(provider_file)
        if not path.exists():
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(path))
        
        # Find __init__ method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check that it has a 'config' parameter
                has_config_param = False
                for arg in node.args.args:
                    if arg.arg == "config":
                        has_config_param = True
                        break
                
                if not has_config_param:
                    violations.append(provider_file)
    
    if violations:
        error_msg = "Providers not using config objects:\n"
        for provider_file in violations:
            error_msg += f"  - {provider_file}\n"
        error_msg += "\nAll providers should accept a config object in __init__."
        pytest.fail(error_msg)


# ============================================================================
# Summary Test
# ============================================================================

def test_infrastructure_import_paths_summary():
    """
    **Summary: Infrastructure import paths are correct**
    
    This test provides a summary of all import path validations.
    It runs all checks and reports any violations found.
    """
    all_violations = []
    
    # Check 1: Deprecated imports
    deprecated = find_deprecated_imports()
    if deprecated:
        all_violations.append(f"Deprecated imports: {len(deprecated)}")
    
    # Check 2: LLM imports
    llm_imports = find_llm_imports_outside_infrastructure()
    if llm_imports:
        all_violations.append(f"Incorrect LLM imports: {len(llm_imports)}")
    
    # Check 3: Cross-domain dependencies
    cross_domain = check_cross_domain_dependencies()
    if cross_domain:
        all_violations.append(f"Cross-domain violations: {len(cross_domain)}")
    
    if all_violations:
        error_msg = "Infrastructure import path violations found:\n"
        for violation in all_violations:
            error_msg += f"  - {violation}\n"
        error_msg += "\nRun individual tests for details."
        pytest.fail(error_msg)
