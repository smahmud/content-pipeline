"""
Property-Based Tests for Provider Naming Conventions

Property 1: File naming follows deployment-service pattern
Property 2: Class naming follows Provider suffix pattern
Validates: Requirements 1.3, 1.4, 2.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6

This test verifies that all provider files and classes follow the established
naming conventions for consistency and maintainability.

Tests both LLM providers and Transcription providers.
"""

import os
import re
import pytest
import inspect
from pathlib import Path
from typing import List, Tuple

# LLM providers
from pipeline.llm.providers import base
from pipeline.llm.providers import local_ollama
from pipeline.llm.providers import cloud_openai
from pipeline.llm.providers import cloud_aws_bedrock
from pipeline.llm.providers import cloud_anthropic

# Transcription providers
from pipeline.transcription.providers import base as transcription_base
from pipeline.transcription.providers import local_whisper
from pipeline.transcription.providers import cloud_openai_whisper
from pipeline.transcription.providers import cloud_aws_transcribe


class TestProviderNamingConventions:
    """Property tests for provider naming conventions."""
    
    def get_provider_files(self, provider_type: str = "llm") -> List[Path]:
        """Get all provider files in the providers directory.
        
        Args:
            provider_type: Type of provider ("llm" or "transcription")
        """
        if provider_type == "llm":
            providers_dir = Path("pipeline/llm/providers")
        elif provider_type == "transcription":
            providers_dir = Path("pipeline/transcription/providers")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        return [
            f for f in providers_dir.glob("*.py")
            if f.name not in ["__init__.py", "base.py"]
        ]
    
    def get_provider_classes(self, module) -> List[Tuple[str, type]]:
        """Get all provider classes from a module."""
        classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only include classes defined in this module
            if obj.__module__ == module.__name__:
                # Exclude base classes and dataclasses
                if name.endswith("Provider") and name != "BaseLLMProvider":
                    classes.append((name, obj))
        return classes
    
    def test_property_all_provider_files_follow_deployment_service_pattern(self):
        """
        Property 1.1: All provider files follow deployment-service naming pattern.
        
        Provider files should be named: {deployment}_{service}.py
        where deployment is 'local' or 'cloud' and service is the service name.
        
        Examples:
        - local_ollama.py (local deployment, ollama service)
        - cloud_openai.py (cloud deployment, openai service)
        - cloud_aws_bedrock.py (cloud deployment, aws_bedrock service)
        - local_whisper.py (local deployment, whisper service)
        - cloud_openai_whisper.py (cloud deployment, openai_whisper service)
        """
        # Pattern: deployment_service.py where deployment is local or cloud
        pattern = re.compile(r'^(local|cloud)_[a-z][a-z0-9_]*\.py$')
        
        # Test LLM providers
        llm_provider_files = self.get_provider_files("llm")
        for file_path in llm_provider_files:
            filename = file_path.name
            assert pattern.match(filename), \
                f"LLM provider file {filename} does not follow deployment_service.py pattern"
        
        # Test Transcription providers
        transcription_provider_files = self.get_provider_files("transcription")
        for file_path in transcription_provider_files:
            filename = file_path.name
            assert pattern.match(filename), \
                f"Transcription provider file {filename} does not follow deployment_service.py pattern"
    
    def test_property_all_provider_classes_end_with_provider(self):
        """
        Property 2.1: All provider classes end with 'Provider' suffix.
        
        Provider class names should follow: {Deployment}{Service}Provider
        where Deployment is 'Local' or 'Cloud' and Service is the service name.
        
        Examples:
        - LocalOllamaProvider
        - CloudOpenAIProvider
        - CloudAWSBedrockProvider
        - LocalWhisperProvider
        - CloudOpenAIWhisperProvider
        """
        # LLM providers
        llm_modules = [local_ollama, cloud_openai, cloud_aws_bedrock, cloud_anthropic]
        
        for module in llm_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, class_obj in classes:
                assert class_name.endswith("Provider"), \
                    f"LLM provider class {class_name} in {module.__name__} does not end with 'Provider'"
        
        # Transcription providers
        transcription_modules = [local_whisper, cloud_openai_whisper, cloud_aws_transcribe]
        
        for module in transcription_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, class_obj in classes:
                assert class_name.endswith("Provider"), \
                    f"Transcription provider class {class_name} in {module.__name__} does not end with 'Provider'"
    
    def test_property_file_name_matches_class_name(self):
        """
        Property 1.2 & 2.2: File name corresponds to class name.
        
        The file name should be the snake_case version of the class name
        (without the Provider suffix).
        
        Examples:
        - local_ollama.py contains LocalOllamaProvider
        - cloud_openai.py contains CloudOpenAIProvider
        - local_whisper.py contains LocalWhisperProvider
        - cloud_openai_whisper.py contains CloudOpenAIWhisperProvider
        """
        # LLM providers
        llm_test_cases = [
            ("local_ollama.py", "LocalOllamaProvider", local_ollama),
            ("cloud_openai.py", "CloudOpenAIProvider", cloud_openai),
            ("cloud_aws_bedrock.py", "CloudAWSBedrockProvider", cloud_aws_bedrock),
            ("cloud_anthropic.py", "CloudAnthropicProvider", cloud_anthropic),
        ]
        
        for filename, expected_class, module in llm_test_cases:
            classes = self.get_provider_classes(module)
            class_names = [name for name, _ in classes]
            
            assert expected_class in class_names, \
                f"Expected LLM provider class {expected_class} not found in {filename}"
        
        # Transcription providers
        transcription_test_cases = [
            ("local_whisper.py", "LocalWhisperProvider", local_whisper),
            ("cloud_openai_whisper.py", "CloudOpenAIWhisperProvider", cloud_openai_whisper),
            ("cloud_aws_transcribe.py", "CloudAWSTranscribeProvider", cloud_aws_transcribe),
        ]
        
        for filename, expected_class, module in transcription_test_cases:
            classes = self.get_provider_classes(module)
            class_names = [name for name, _ in classes]
            
            assert expected_class in class_names, \
                f"Expected transcription provider class {expected_class} not found in {filename}"
    
    def test_property_local_providers_start_with_local(self):
        """
        Property 1.3: Local deployment providers start with 'local_'.
        
        Files for local deployment should start with 'local_' prefix.
        """
        # Test LLM providers
        llm_provider_files = self.get_provider_files("llm")
        llm_local_files = [f for f in llm_provider_files if f.name.startswith("local_")]
        
        # Verify we have at least one local LLM provider
        assert len(llm_local_files) > 0, "Should have at least one local LLM provider"
        
        # Verify all local files follow pattern
        for file_path in llm_local_files:
            expected_class_prefix = "Local"
            content = file_path.read_text()
            assert f"class {expected_class_prefix}" in content, \
                f"Local LLM provider file {file_path.name} should contain class starting with '{expected_class_prefix}'"
        
        # Test Transcription providers
        transcription_provider_files = self.get_provider_files("transcription")
        transcription_local_files = [f for f in transcription_provider_files if f.name.startswith("local_")]
        
        # Verify we have at least one local transcription provider
        assert len(transcription_local_files) > 0, "Should have at least one local transcription provider"
        
        # Verify all local files follow pattern
        for file_path in transcription_local_files:
            expected_class_prefix = "Local"
            content = file_path.read_text()
            assert f"class {expected_class_prefix}" in content, \
                f"Local transcription provider file {file_path.name} should contain class starting with '{expected_class_prefix}'"
    
    def test_property_cloud_providers_start_with_cloud(self):
        """
        Property 1.4: Cloud deployment providers start with 'cloud_'.
        
        Files for cloud deployment should start with 'cloud_' prefix.
        """
        # Test LLM providers
        llm_provider_files = self.get_provider_files("llm")
        llm_cloud_files = [f for f in llm_provider_files if f.name.startswith("cloud_")]
        
        # Verify we have cloud LLM providers
        assert len(llm_cloud_files) > 0, "Should have at least one cloud LLM provider"
        
        # Verify all cloud files follow pattern
        for file_path in llm_cloud_files:
            expected_class_prefix = "Cloud"
            content = file_path.read_text()
            assert f"class {expected_class_prefix}" in content, \
                f"Cloud LLM provider file {file_path.name} should contain class starting with '{expected_class_prefix}'"
        
        # Test Transcription providers
        transcription_provider_files = self.get_provider_files("transcription")
        transcription_cloud_files = [f for f in transcription_provider_files if f.name.startswith("cloud_")]
        
        # Verify we have cloud transcription providers
        assert len(transcription_cloud_files) > 0, "Should have at least one cloud transcription provider"
        
        # Verify all cloud files follow pattern
        for file_path in transcription_cloud_files:
            expected_class_prefix = "Cloud"
            content = file_path.read_text()
            assert f"class {expected_class_prefix}" in content, \
                f"Cloud transcription provider file {file_path.name} should contain class starting with '{expected_class_prefix}'"
    
    def test_property_no_agent_terminology_in_files(self):
        """
        Property 2.3: No 'Agent' or 'Adapter' terminology in provider files.
        
        All files should use 'Provider' terminology, not 'Agent' or 'Adapter'.
        """
        # Test LLM providers
        llm_provider_files = self.get_provider_files("llm")
        
        for file_path in llm_provider_files:
            # Check filename doesn't contain 'agent' or 'adapter'
            assert 'agent' not in file_path.name.lower(), \
                f"LLM provider file {file_path.name} contains 'agent' terminology"
            assert 'adapter' not in file_path.name.lower(), \
                f"LLM provider file {file_path.name} contains 'adapter' terminology"
            
            # Check file content
            content = file_path.read_text()
            
            # Allow 'agent' or 'adapter' in comments or docstrings explaining migration
            # but not in class names or variable names
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                
                # Check for Agent/Adapter in class names
                if 'class ' in line and ('Agent' in line or 'Adapter' in line) and 'Provider' not in line:
                    pytest.fail(
                        f"LLM provider file {file_path.name} line {i} contains 'Agent' or 'Adapter' class name: {line.strip()}"
                    )
        
        # Test Transcription providers
        transcription_provider_files = self.get_provider_files("transcription")
        
        for file_path in transcription_provider_files:
            # Check filename doesn't contain 'agent' or 'adapter'
            assert 'agent' not in file_path.name.lower(), \
                f"Transcription provider file {file_path.name} contains 'agent' terminology"
            assert 'adapter' not in file_path.name.lower(), \
                f"Transcription provider file {file_path.name} contains 'adapter' terminology"
            
            # Check file content
            content = file_path.read_text()
            
            # Allow 'agent' or 'adapter' in comments or docstrings explaining migration
            # but not in class names or variable names
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                
                # Check for Agent/Adapter in class names
                if 'class ' in line and ('Agent' in line or 'Adapter' in line) and 'Provider' not in line:
                    pytest.fail(
                        f"Transcription provider file {file_path.name} line {i} contains 'Agent' or 'Adapter' class name: {line.strip()}"
                    )
    
    def test_property_all_providers_inherit_from_base(self):
        """
        Property 2.4: All provider classes inherit from base provider class.
        
        Every LLM provider should inherit from BaseLLMProvider.
        Every Transcription provider should implement TranscriberProvider protocol.
        """
        # LLM providers
        llm_modules = [local_ollama, cloud_openai, cloud_aws_bedrock, cloud_anthropic]
        
        for module in llm_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, class_obj in classes:
                # Check if it inherits from BaseLLMProvider
                bases = [base.__name__ for base in class_obj.__bases__]
                assert "BaseLLMProvider" in bases, \
                    f"LLM provider class {class_name} does not inherit from BaseLLMProvider"
        
        # Transcription providers - they implement a Protocol, not inherit from a base class
        # We can check if they have the required methods
        transcription_modules = [local_whisper, cloud_openai_whisper, cloud_aws_transcribe]
        required_methods = ['transcribe', 'get_engine_info', 'validate_requirements', 
                          'get_supported_formats', 'estimate_cost']
        
        for module in transcription_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, class_obj in classes:
                # Check if it has all required methods from TranscriberProvider protocol
                for method_name in required_methods:
                    assert hasattr(class_obj, method_name), \
                        f"Transcription provider class {class_name} does not implement required method '{method_name}'"
    
    def test_property_provider_files_in_correct_directory(self):
        """
        Property 1.5: All provider files are in correct providers/ directory.
        
        LLM provider files should be in pipeline/llm/providers/ directory.
        Transcription provider files should be in pipeline/transcription/providers/ directory.
        """
        # Test LLM providers
        llm_providers_dir = Path("pipeline/llm/providers")
        assert llm_providers_dir.exists(), "LLM providers directory should exist"
        assert llm_providers_dir.is_dir(), "LLM providers path should be a directory"
        
        llm_provider_files = self.get_provider_files("llm")
        assert len(llm_provider_files) > 0, "Should have at least one LLM provider file"
        
        for file_path in llm_provider_files:
            assert file_path.parent.name == "providers", \
                f"LLM provider file {file_path.name} is not in providers directory"
            assert "pipeline/llm/providers" in file_path.as_posix(), \
                f"LLM provider file {file_path.name} is not in correct path"
        
        # Test Transcription providers
        transcription_providers_dir = Path("pipeline/transcription/providers")
        assert transcription_providers_dir.exists(), "Transcription providers directory should exist"
        assert transcription_providers_dir.is_dir(), "Transcription providers path should be a directory"
        
        transcription_provider_files = self.get_provider_files("transcription")
        assert len(transcription_provider_files) > 0, "Should have at least one transcription provider file"
        
        for file_path in transcription_provider_files:
            assert file_path.parent.name == "providers", \
                f"Transcription provider file {file_path.name} is not in providers directory"
            assert "pipeline/transcription/providers" in file_path.as_posix(), \
                f"Transcription provider file {file_path.name} is not in correct path"
    
    def test_property_consistent_naming_across_deployment_types(self):
        """
        Property 1.6 & 2.5: Naming is consistent across deployment types.
        
        The pattern should be consistent: {deployment}_{service}.py
        contains {Deployment}{Service}Provider class.
        """
        # LLM providers
        llm_test_cases = [
            ("local", "ollama", "LocalOllamaProvider", "pipeline/llm/providers"),
            ("cloud", "openai", "CloudOpenAIProvider", "pipeline/llm/providers"),
            ("cloud", "aws_bedrock", "CloudAWSBedrockProvider", "pipeline/llm/providers"),
            ("cloud", "anthropic", "CloudAnthropicProvider", "pipeline/llm/providers"),
        ]
        
        for deployment, service, expected_class, base_path in llm_test_cases:
            # Check file exists
            filename = f"{deployment}_{service}.py"
            file_path = Path(base_path) / filename
            assert file_path.exists(), f"Expected LLM provider file {filename} does not exist"
            
            # Check class exists in file
            content = file_path.read_text()
            assert f"class {expected_class}" in content, \
                f"Expected LLM provider class {expected_class} not found in {filename}"
        
        # Transcription providers
        transcription_test_cases = [
            ("local", "whisper", "LocalWhisperProvider", "pipeline/transcription/providers"),
            ("cloud", "openai_whisper", "CloudOpenAIWhisperProvider", "pipeline/transcription/providers"),
            ("cloud", "aws_transcribe", "CloudAWSTranscribeProvider", "pipeline/transcription/providers"),
        ]
        
        for deployment, service, expected_class, base_path in transcription_test_cases:
            # Check file exists
            filename = f"{deployment}_{service}.py"
            file_path = Path(base_path) / filename
            assert file_path.exists(), f"Expected transcription provider file {filename} does not exist"
            
            # Check class exists in file
            content = file_path.read_text()
            assert f"class {expected_class}" in content, \
                f"Expected transcription provider class {expected_class} not found in {filename}"
    
    def test_property_no_duplicate_provider_names(self):
        """
        Property 2.6: No duplicate provider class names.
        
        Each provider should have a unique class name across all provider types.
        """
        all_classes = []
        
        # LLM providers
        llm_modules = [local_ollama, cloud_openai, cloud_aws_bedrock, cloud_anthropic]
        
        for module in llm_modules:
            classes = self.get_provider_classes(module)
            for class_name, _ in classes:
                all_classes.append(("llm", class_name))
        
        # Transcription providers
        transcription_modules = [local_whisper, cloud_openai_whisper, cloud_aws_transcribe]
        
        for module in transcription_modules:
            classes = self.get_provider_classes(module)
            for class_name, _ in classes:
                all_classes.append(("transcription", class_name))
        
        # Check for duplicates
        class_names_only = [name for _, name in all_classes]
        assert len(class_names_only) == len(set(class_names_only)), \
            f"Duplicate provider class names found: {class_names_only}"
    
    def test_property_provider_files_use_snake_case(self):
        """
        Property 1.7: Provider files use snake_case naming.
        
        All provider files should use lowercase with underscores.
        """
        # Pattern: only lowercase letters, numbers, and underscores
        pattern = re.compile(r'^[a-z][a-z0-9_]*\.py$')
        
        # Test LLM providers
        llm_provider_files = self.get_provider_files("llm")
        
        for file_path in llm_provider_files:
            filename = file_path.name
            assert pattern.match(filename), \
                f"LLM provider file {filename} does not use snake_case naming"
            
            # Verify no uppercase letters
            assert filename == filename.lower(), \
                f"LLM provider file {filename} contains uppercase letters"
        
        # Test Transcription providers
        transcription_provider_files = self.get_provider_files("transcription")
        
        for file_path in transcription_provider_files:
            filename = file_path.name
            assert pattern.match(filename), \
                f"Transcription provider file {filename} does not use snake_case naming"
            
            # Verify no uppercase letters
            assert filename == filename.lower(), \
                f"Transcription provider file {filename} contains uppercase letters"
    
    def test_property_provider_classes_use_pascal_case(self):
        """
        Property 2.7: Provider classes use PascalCase naming.
        
        All provider class names should start with uppercase and use PascalCase.
        """
        # LLM providers
        llm_modules = [local_ollama, cloud_openai, cloud_aws_bedrock, cloud_anthropic]
        
        for module in llm_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, _ in classes:
                # Check starts with uppercase
                assert class_name[0].isupper(), \
                    f"LLM provider class {class_name} does not start with uppercase letter"
                
                # Check no underscores (PascalCase, not snake_case)
                assert '_' not in class_name, \
                    f"LLM provider class {class_name} contains underscores (should use PascalCase)"
        
        # Transcription providers
        transcription_modules = [local_whisper, cloud_openai_whisper, cloud_aws_transcribe]
        
        for module in transcription_modules:
            classes = self.get_provider_classes(module)
            
            for class_name, _ in classes:
                # Check starts with uppercase
                assert class_name[0].isupper(), \
                    f"Transcription provider class {class_name} does not start with uppercase letter"
                
                # Check no underscores (PascalCase, not snake_case)
                assert '_' not in class_name, \
                    f"Transcription provider class {class_name} contains underscores (should use PascalCase)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
