"""
Unit tests for CodeSampleGenerator.

Tests technical content detection, language detection, and code generation.
"""

import pytest

from pipeline.formatters.code_samples import (
    CodeSample,
    CodeSampleGenerator,
    CodeSamplesResult,
    LANGUAGE_MAPPINGS,
    SUPPORTED_OUTPUT_TYPES,
    TECHNICAL_INDICATORS,
    UNSUPPORTED_OUTPUT_TYPES,
)


class TestCodeSampleDataclass:
    """Tests for CodeSample dataclass."""
    
    def test_create_code_sample(self):
        """Test basic CodeSample creation."""
        sample = CodeSample(
            code="print('hello')",
            language="python",
            description="A simple print statement",
            position_hint="introduction"
        )
        assert sample.code == "print('hello')"
        assert sample.language == "python"
        assert sample.description == "A simple print statement"
        assert sample.position_hint == "introduction"


class TestCodeSamplesResultDataclass:
    """Tests for CodeSamplesResult dataclass."""
    
    def test_create_result(self):
        """Test basic CodeSamplesResult creation."""
        result = CodeSamplesResult(
            samples=[],
            is_technical=True,
            detected_topics=["python"],
            detected_languages=["python"]
        )
        assert result.samples == []
        assert result.is_technical is True
        assert result.detected_topics == ["python"]
        assert result.detected_languages == ["python"]
    
    def test_result_default_values(self):
        """Test CodeSamplesResult with default values."""
        result = CodeSamplesResult(
            samples=[],
            is_technical=False
        )
        assert result.detected_topics == []
        assert result.detected_languages == []


class TestCodeSampleGeneratorSupport:
    """Tests for output type support checking."""
    
    def test_supported_output_types(self):
        """Test that supported types return True."""
        generator = CodeSampleGenerator()
        
        for output_type in SUPPORTED_OUTPUT_TYPES:
            assert generator.is_supported(output_type), f"{output_type} should be supported"
    
    def test_unsupported_output_types(self):
        """Test that unsupported types return False."""
        generator = CodeSampleGenerator()
        
        for output_type in UNSUPPORTED_OUTPUT_TYPES:
            assert not generator.is_supported(output_type), f"{output_type} should not be supported"
    
    def test_unknown_output_type(self):
        """Test that unknown types return False."""
        generator = CodeSampleGenerator()
        assert not generator.is_supported("unknown-type")


class TestTechnicalContentDetection:
    """Tests for technical content detection."""
    
    def test_detect_technical_by_topics(self):
        """Test detection via topics."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["python", "web development", "api"],
        }
        
        assert generator.is_technical_content(content) is True
    
    def test_detect_technical_by_tags(self):
        """Test detection via tags."""
        generator = CodeSampleGenerator()
        
        content = {
            "tags": ["programming", "tutorial", "code"],
        }
        
        assert generator.is_technical_content(content) is True
    
    def test_detect_technical_by_tags_dict(self):
        """Test detection via tags as dict."""
        generator = CodeSampleGenerator()
        
        content = {
            "tags": {
                "primary": ["javascript", "react"],
                "secondary": ["frontend"],
            },
        }
        
        assert generator.is_technical_content(content) is True
    
    def test_detect_technical_by_summary(self):
        """Test detection via summary."""
        generator = CodeSampleGenerator()
        
        content = {
            "summary": {
                "short": "Learn how to build REST APIs with Python",
            },
        }
        
        assert generator.is_technical_content(content) is True
    
    def test_detect_technical_by_title(self):
        """Test detection via title."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {
                "title": "Docker Tutorial for Beginners",
            },
        }
        
        assert generator.is_technical_content(content) is True
    
    def test_non_technical_content(self):
        """Test non-technical content returns False."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["cooking", "recipes", "food"],
            "tags": ["lifestyle", "health"],
            "summary": {"short": "How to make delicious pasta"},
        }
        
        assert generator.is_technical_content(content) is False
    
    def test_empty_content(self):
        """Test empty content returns False."""
        generator = CodeSampleGenerator()
        
        assert generator.is_technical_content({}) is False


class TestLanguageDetection:
    """Tests for programming language detection."""
    
    def test_detect_python(self):
        """Test Python detection."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["python", "django"],
        }
        
        languages = generator.detect_languages(content)
        assert "python" in languages
    
    def test_detect_javascript(self):
        """Test JavaScript detection."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["javascript", "node"],
        }
        
        languages = generator.detect_languages(content)
        assert "javascript" in languages
    
    def test_detect_typescript(self):
        """Test TypeScript detection."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["typescript", "react"],
        }
        
        languages = generator.detect_languages(content)
        assert "typescript" in languages
    
    def test_detect_bash_from_cli(self):
        """Test Bash detection from CLI topics."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["cli", "command-line", "terminal"],
        }
        
        languages = generator.detect_languages(content)
        assert "bash" in languages
    
    def test_detect_sql_from_database(self):
        """Test SQL detection from database topics."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["database", "sql", "query"],
        }
        
        languages = generator.detect_languages(content)
        assert "sql" in languages
    
    def test_detect_multiple_languages(self):
        """Test detection of multiple languages."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["python", "javascript", "docker"],
        }
        
        languages = generator.detect_languages(content)
        assert len(languages) >= 2
        assert "python" in languages
        assert "javascript" in languages
    
    def test_default_to_python_for_technical(self):
        """Test default to Python for generic technical content."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["programming", "software development"],
        }
        
        languages = generator.detect_languages(content)
        assert "python" in languages
    
    def test_no_language_for_non_technical(self):
        """Test no language detected for non-technical content."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["cooking", "recipes"],
        }
        
        languages = generator.detect_languages(content)
        assert len(languages) == 0


class TestCodeSampleGeneration:
    """Tests for code sample generation."""
    
    def test_generate_for_python_content(self):
        """Test generating samples for Python content."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {"title": "Python Tutorial"},
            "topics": ["python", "programming"],
            "tags": ["python", "tutorial"],
        }
        
        result = generator.generate(content, "blog")
        
        assert result.is_technical is True
        assert len(result.samples) > 0
        assert "python" in result.detected_languages
        
        # Check sample has required fields
        sample = result.samples[0]
        assert sample.code
        assert sample.language
        assert sample.description
        assert sample.position_hint
    
    def test_generate_for_cli_content(self):
        """Test generating samples for CLI content."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {"title": "Bash Scripting Guide"},
            "topics": ["cli", "bash", "terminal"],
        }
        
        result = generator.generate(content, "tutorial")
        
        assert result.is_technical is True
        assert "bash" in result.detected_languages
    
    def test_generate_for_web_content(self):
        """Test generating samples for web content."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {"title": "React Tutorial"},
            "topics": ["react", "frontend", "typescript"],
        }
        
        result = generator.generate(content, "blog")
        
        assert result.is_technical is True
        assert "typescript" in result.detected_languages
    
    def test_skip_non_technical_content(self):
        """Test that non-technical content returns empty samples."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {"title": "Cooking Tips"},
            "topics": ["cooking", "recipes"],
        }
        
        result = generator.generate(content, "blog")
        
        assert result.is_technical is False
        assert len(result.samples) == 0
    
    def test_skip_unsupported_output_type(self):
        """Test that unsupported output types return empty samples."""
        generator = CodeSampleGenerator()
        
        content = {
            "metadata": {"title": "Python Tutorial"},
            "topics": ["python", "programming"],
        }
        
        result = generator.generate(content, "tweet")
        
        assert len(result.samples) == 0


class TestCodeSampleFormatting:
    """Tests for code sample formatting."""
    
    def test_format_for_markdown(self):
        """Test formatting sample for Markdown."""
        generator = CodeSampleGenerator()
        
        sample = CodeSample(
            code="print('hello')",
            language="python",
            description="A simple print statement",
            position_hint="introduction"
        )
        
        formatted = generator.format_for_markdown(sample)
        
        assert "```python" in formatted
        assert "print('hello')" in formatted
        assert "```" in formatted
        assert "A simple print statement" in formatted


class TestCodeSampleEmbeddingFormat:
    """Tests for code sample embedding format (Property 7)."""
    
    def test_samples_have_markdown_code_fence(self):
        """Test that samples can be formatted with code fences."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["python", "programming"],
        }
        
        result = generator.generate(content, "blog")
        
        for sample in result.samples:
            formatted = generator.format_for_markdown(sample)
            # Should have opening code fence with language
            assert f"```{sample.language}" in formatted
            # Should have closing code fence
            assert formatted.count("```") >= 2
    
    def test_samples_have_comments(self):
        """Test that generated samples include comments."""
        generator = CodeSampleGenerator()
        
        content = {
            "topics": ["python", "programming"],
        }
        
        result = generator.generate(content, "blog")
        
        for sample in result.samples:
            # Python comments start with #
            # JavaScript/TypeScript comments start with //
            has_comment = "#" in sample.code or "//" in sample.code or "/*" in sample.code
            assert has_comment, f"Sample for {sample.language} should have comments"


class TestTechnicalIndicators:
    """Tests for technical indicator constants."""
    
    def test_indicators_are_lowercase(self):
        """Test that indicators are lowercase for matching."""
        for indicator in TECHNICAL_INDICATORS:
            assert indicator == indicator.lower(), f"Indicator '{indicator}' should be lowercase"
    
    def test_common_indicators_present(self):
        """Test that common technical terms are present."""
        common_terms = ["programming", "api", "database", "python", "javascript"]
        for term in common_terms:
            assert term in TECHNICAL_INDICATORS, f"'{term}' should be in indicators"


class TestLanguageMappings:
    """Tests for language mapping constants."""
    
    def test_cli_maps_to_bash(self):
        """Test CLI-related terms map to bash."""
        cli_terms = ["cli", "command-line", "terminal", "shell"]
        for term in cli_terms:
            assert LANGUAGE_MAPPINGS.get(term) == "bash", f"'{term}' should map to bash"
    
    def test_web_maps_to_typescript(self):
        """Test web-related terms map to TypeScript."""
        web_terms = ["frontend", "react", "vue", "angular"]
        for term in web_terms:
            assert LANGUAGE_MAPPINGS.get(term) == "typescript", f"'{term}' should map to typescript"
    
    def test_backend_maps_to_python(self):
        """Test backend-related terms map to Python."""
        backend_terms = ["backend", "api", "django", "flask"]
        for term in backend_terms:
            assert LANGUAGE_MAPPINGS.get(term) == "python", f"'{term}' should map to python"
