"""
Unit tests for Configuration Pretty Printer.

Tests the enhanced YAML formatting with comments, examples, and templates.
"""

import pytest
from pipeline.config.pretty_printer import ConfigurationPrettyPrinter
from pipeline.config.schema import TranscriptionConfig, EngineType


class TestConfigurationPrettyPrinter:
    """Test suite for ConfigurationPrettyPrinter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.printer = ConfigurationPrettyPrinter()
    
    def test_generate_full_template_with_examples(self):
        """Test generating full template with usage examples."""
        template = self.printer.generate_full_template(include_examples=True)
        
        assert "Content Pipeline Configuration v0.6.5" in template
        assert "CORE CONFIGURATION" in template
        assert "LOCAL WHISPER CONFIGURATION" in template
        assert "OPENAI WHISPER API CONFIGURATION" in template
        assert "AWS TRANSCRIBE CONFIGURATION" in template
        assert "AUTO-SELECTION PREFERENCES" in template
        assert "USAGE EXAMPLES" in template
        assert "content-pipeline transcribe" in template
        assert "export OPENAI_API_KEY" in template
    
    def test_generate_full_template_without_examples(self):
        """Test generating full template without usage examples."""
        template = self.printer.generate_full_template(include_examples=False)
        
        assert "Content Pipeline Configuration v0.6.5" in template
        assert "CORE CONFIGURATION" in template
        assert "USAGE EXAMPLES" not in template
        assert "content-pipeline transcribe" not in template
    
    def test_generate_minimal_template(self):
        """Test generating minimal configuration template."""
        template = self.printer.generate_minimal_template()
        
        assert "Content Pipeline Configuration v0.6.5" in template
        assert "Minimal configuration template" in template
        assert "engine: auto" in template
        assert "output_dir: ./transcripts" in template
        assert "whisper_local:" in template
        assert "whisper_api:" in template
        assert "aws_transcribe:" in template
        
        # Should not have extensive comments
        assert "CORE CONFIGURATION" not in template
        assert "Options:" not in template
    
    def test_generate_whisper_local_template(self):
        """Test generating template optimized for local Whisper."""
        template = self.printer.generate_engine_specific_template(EngineType.WHISPER_LOCAL.value)
        
        assert "Optimized for Local Whisper Processing" in template
        assert "engine: whisper-local" in template
        assert "prefer_local: true" in template
        assert "fallback_enabled: false" in template
        assert "Use local Whisper for privacy" in template
    
    def test_generate_whisper_api_template(self):
        """Test generating template optimized for OpenAI API."""
        template = self.printer.generate_engine_specific_template(EngineType.WHISPER_API.value)
        
        assert "Optimized for OpenAI Whisper API" in template
        assert "engine: whisper-api" in template
        assert "prefer_local: false" in template
        assert "fallback_enabled: true" in template
        assert "whisper-api" in template
        assert "whisper-local" in template  # Fallback option
    
    def test_generate_aws_transcribe_template(self):
        """Test generating template optimized for AWS Transcribe."""
        template = self.printer.generate_engine_specific_template(EngineType.AWS_TRANSCRIBE.value)
        
        assert "Optimized for AWS Transcribe Service" in template
        assert "engine: aws-transcribe" in template
        assert "prefer_local: false" in template
        assert "fallback_enabled: true" in template
        assert "aws-transcribe" in template
        assert "whisper-local" in template  # Fallback option
    
    def test_generate_unknown_engine_template(self):
        """Test generating template for unknown engine returns minimal template."""
        template = self.printer.generate_engine_specific_template("unknown-engine")
        
        # Should return minimal template as fallback
        assert "Minimal configuration template" in template
        assert "engine: auto" in template
    
    def test_format_configuration_full_style(self):
        """Test formatting configuration with full style."""
        config = TranscriptionConfig(
            engine="whisper-local",
            output_dir="./test-output",
            log_level="debug"
        )
        
        formatted = self.printer.format_configuration(config, style="full")
        
        assert "Content Pipeline Configuration v0.6.5" in formatted
        assert "Generated from current settings" in formatted
        assert "engine: whisper-local" in formatted
        assert "output_dir: ./test-output" in formatted
        assert "log_level: debug" in formatted
        assert "# Transcription engine" in formatted
        assert "whisper_local:" in formatted
        assert "whisper_api:" in formatted
        assert "aws_transcribe:" in formatted
    
    def test_format_configuration_minimal_style(self):
        """Test formatting configuration with minimal style."""
        config = TranscriptionConfig(
            engine="whisper-api",
            output_dir="./api-output"
        )
        
        formatted = self.printer.format_configuration(config, style="minimal")
        
        assert "Content Pipeline Configuration v0.6.5" in formatted
        assert "engine: whisper-api" in formatted
        assert "output_dir: ./api-output" in formatted
        assert "whisper_local:" in formatted
        assert "whisper_api:" in formatted
        
        # Should have minimal comments
        assert "Generated from current settings" not in formatted
        assert "# Transcription engine" not in formatted
    
    def test_format_configuration_compact_style(self):
        """Test formatting configuration with compact style."""
        config = TranscriptionConfig(
            engine="aws-transcribe",
            log_level="warning"
        )
        
        formatted = self.printer.format_configuration(config, style="compact")
        
        # Should only include non-default values
        assert "engine: aws-transcribe" in formatted
        assert "log_level: warning" in formatted
        
        # Should not include default values or comments
        assert "Content Pipeline Configuration" not in formatted
        assert "output_dir: ./transcripts" not in formatted  # Default value
    
    def test_format_configuration_default_style(self):
        """Test formatting configuration with default/unknown style."""
        config = TranscriptionConfig()
        
        formatted = self.printer.format_configuration(config, style="unknown")
        
        # Should default to full style
        assert "Content Pipeline Configuration v0.6.5" in formatted
        assert "Generated from current settings" in formatted
    
    def test_format_configuration_with_environment_variables(self):
        """Test that environment variable placeholders are preserved."""
        config = TranscriptionConfig()
        
        formatted = self.printer.format_configuration(config, style="full")
        
        assert "${OPENAI_API_KEY}" in formatted
        assert "${AWS_ACCESS_KEY_ID}" in formatted
        assert "${AWS_SECRET_ACCESS_KEY}" in formatted
    
    def test_format_configuration_with_actual_values(self):
        """Test formatting configuration with actual API key values."""
        config = TranscriptionConfig()
        config.whisper_api.api_key = "sk-actual-key"
        config.aws_transcribe.access_key_id = "AKIA-actual-key"
        
        formatted = self.printer.format_configuration(config, style="full")
        
        assert "api_key: sk-actual-key" in formatted
        assert "access_key_id: AKIA-actual-key" in formatted
        # Should still use placeholder for unset values
        assert "${AWS_SECRET_ACCESS_KEY}" in formatted
    
    def test_all_templates_are_valid_yaml(self):
        """Test that all generated templates are valid YAML."""
        import yaml
        
        templates = [
            self.printer.generate_full_template(include_examples=True),
            self.printer.generate_full_template(include_examples=False),
            self.printer.generate_minimal_template(),
            self.printer.generate_engine_specific_template("whisper-local"),
            self.printer.generate_engine_specific_template("whisper-api"),
            self.printer.generate_engine_specific_template("aws-transcribe"),
        ]
        
        for template in templates:
            # Remove comments and examples for YAML parsing
            lines = template.split('\n')
            yaml_lines = []
            
            for line in lines:
                # Skip comment-only lines and example sections
                if (line.strip().startswith('#') or 
                    'content-pipeline transcribe' in line or
                    'export ' in line):
                    continue
                yaml_lines.append(line)
            
            yaml_content = '\n'.join(yaml_lines)
            
            # Should parse without errors
            try:
                parsed = yaml.safe_load(yaml_content)
                assert parsed is not None or yaml_content.strip() == ""
            except yaml.YAMLError as e:
                pytest.fail(f"Generated template is not valid YAML: {e}\nTemplate:\n{yaml_content}")
    
    def test_template_contains_all_required_sections(self):
        """Test that full template contains all required configuration sections."""
        template = self.printer.generate_full_template()
        
        required_sections = [
            'engine:',
            'output_dir:',
            'log_level:',
            'language:',
            'whisper_local:',
            'whisper_api:',
            'aws_transcribe:',
            'auto_selection:'
        ]
        
        for section in required_sections:
            assert section in template, f"Template missing required section: {section}"