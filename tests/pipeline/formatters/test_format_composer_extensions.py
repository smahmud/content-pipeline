"""
Unit tests for FormatComposer extensions.

Tests format_from_sources, generate_image_prompts, and generate_code_samples
methods added in v0.8.7.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.formatters.composer import FormatComposer
from pipeline.formatters.source_combiner import SourceCombiner, SourceFile, CombinedContent
from pipeline.formatters.image_prompts import (
    ImagePromptGenerator,
    ImagePrompt,
    ImagePromptsResult,
)
from pipeline.formatters.code_samples import (
    CodeSampleGenerator,
    CodeSample,
    CodeSamplesResult,
)


@pytest.fixture
def enriched_content():
    """Sample enriched content."""
    return {
        "enrichment_version": "v1",
        "metadata": {
            "title": "Backend Developer Roadmap",
            "source_file": "test.json",
            "provider": "openai",
            "model": "gpt-4",
            "timestamp": "2026-02-20T12:00:00Z",
            "cost_usd": 0.05,
            "tokens_used": 1500,
            "enrichment_types": ["summary", "tags", "highlights"],
        },
        "summary": {
            "short": "A guide to backend development.",
            "medium": "A comprehensive guide covering APIs, databases, and servers.",
            "long": "A comprehensive guide to backend development covering REST APIs, databases, server management, and deployment.",
        },
        "topics": ["backend", "api", "database", "python"],
        "tags": ["programming", "backend", "tutorial"],
        "highlights": [
            {"text": "REST APIs are the backbone of web services"},
            {"text": "Database design is critical for performance"},
        ],
    }


@pytest.fixture
def composer():
    """Create a FormatComposer with mocked LLM."""
    return FormatComposer(llm_enhancer=None)


class TestFormatComposerInit:
    """Tests for extended FormatComposer initialization."""

    def test_default_source_combiner(self):
        """Test SourceCombiner is created by default."""
        composer = FormatComposer()
        assert composer.source_combiner is not None
        assert isinstance(composer.source_combiner, SourceCombiner)

    def test_custom_source_combiner(self):
        """Test custom SourceCombiner is used."""
        custom = SourceCombiner()
        composer = FormatComposer(source_combiner=custom)
        assert composer.source_combiner is custom

    def test_image_prompt_generator_none_by_default(self):
        """Test ImagePromptGenerator is None by default."""
        composer = FormatComposer()
        assert composer.image_prompt_generator is None

    def test_custom_image_prompt_generator(self):
        """Test custom ImagePromptGenerator is used."""
        custom = ImagePromptGenerator()
        composer = FormatComposer(image_prompt_generator=custom)
        assert composer.image_prompt_generator is custom

    def test_code_sample_generator_none_by_default(self):
        """Test CodeSampleGenerator is None by default."""
        composer = FormatComposer()
        assert composer.code_sample_generator is None

    def test_custom_code_sample_generator(self):
        """Test custom CodeSampleGenerator is used."""
        custom = CodeSampleGenerator()
        composer = FormatComposer(code_sample_generator=custom)
        assert composer.code_sample_generator is custom


class TestFormatFromSources:
    """Tests for format_from_sources method."""

    def test_empty_folder_returns_failure(self, composer, tmp_path):
        """Test empty folder returns failure result."""
        result = composer.format_from_sources(
            sources_folder=tmp_path,
            output_type="blog",
        )
        assert result.success is False
        assert "No supported files" in result.error

    def test_folder_with_enriched_json(self, composer, tmp_path, enriched_content):
        """Test formatting from folder with enriched JSON."""
        # Create a source file
        source_file = tmp_path / "test-enriched.json"
        source_file.write_text(json.dumps(enriched_content), encoding="utf-8")

        result = composer.format_from_sources(
            sources_folder=tmp_path,
            output_type="blog",
            llm_enhance=False,
        )
        assert result.success is True
        assert len(result.content) > 0

    def test_format_from_sources_passes_options(self, composer, tmp_path, enriched_content):
        """Test that options are passed through to format_single."""
        source_file = tmp_path / "test-enriched.json"
        source_file.write_text(json.dumps(enriched_content), encoding="utf-8")

        result = composer.format_from_sources(
            sources_folder=tmp_path,
            output_type="blog",
            platform="medium",
            tone="casual",
            length="short",
            llm_enhance=False,
        )
        assert result.success is True

    def test_nonexistent_folder_returns_failure(self, composer):
        """Test nonexistent folder returns failure."""
        result = composer.format_from_sources(
            sources_folder=Path("/nonexistent/folder"),
            output_type="blog",
        )
        assert result.success is False


class TestGenerateImagePrompts:
    """Tests for generate_image_prompts method."""

    def test_creates_generator_if_none(self, composer, enriched_content):
        """Test generator is created on first call."""
        assert composer.image_prompt_generator is None
        composer.generate_image_prompts(enriched_content, "blog")
        assert composer.image_prompt_generator is not None

    def test_returns_prompts_for_supported_type(self, composer, enriched_content):
        """Test returns prompts for supported output type."""
        result = composer.generate_image_prompts(enriched_content, "blog")
        assert result is not None
        assert len(result.prompts) > 0

    def test_returns_none_for_unsupported_type(self, composer, enriched_content):
        """Test returns None for unsupported output type."""
        result = composer.generate_image_prompts(enriched_content, "tweet")
        assert result is None

    def test_passes_platform(self, composer, enriched_content):
        """Test platform is passed to generator."""
        result = composer.generate_image_prompts(
            enriched_content, "blog", platform="medium"
        )
        assert result is not None
        assert result.platform == "medium"


class TestGenerateCodeSamples:
    """Tests for generate_code_samples method."""

    def test_creates_generator_if_none(self, composer, enriched_content):
        """Test generator is created on first call."""
        assert composer.code_sample_generator is None
        composer.generate_code_samples(enriched_content, "blog")
        assert composer.code_sample_generator is not None

    def test_returns_samples_for_technical_content(self, composer, enriched_content):
        """Test returns samples for technical content."""
        result = composer.generate_code_samples(enriched_content, "blog")
        assert result is not None
        assert len(result.samples) > 0

    def test_returns_none_for_unsupported_type(self, composer, enriched_content):
        """Test returns None for unsupported output type."""
        result = composer.generate_code_samples(enriched_content, "tweet")
        assert result is None

    def test_returns_none_for_non_technical(self, composer):
        """Test returns None for non-technical content."""
        non_tech = {
            "summary": {"short": "A cooking recipe for pasta."},
            "topics": ["cooking", "food", "recipes"],
            "tags": ["lifestyle"],
        }
        result = composer.generate_code_samples(non_tech, "blog")
        assert result is None
