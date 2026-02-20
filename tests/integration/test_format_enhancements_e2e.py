"""
Minimal end-to-end integration test for v0.8.7 format command enhancements.

Tests the full flow: source files → SourceCombiner → FormatComposer → output.
"""

import json
import pytest
from pathlib import Path

from pipeline.formatters.composer import FormatComposer
from pipeline.formatters.source_combiner import SourceCombiner


@pytest.fixture
def enriched_source_folder(tmp_path):
    """Create a temp folder with enriched JSON files for multi-source testing."""
    # Source 1: enriched JSON
    source1 = {
        "enrichment_version": "v1",
        "metadata": {
            "title": "REST API Design",
            "source_file": "rest-api.json",
            "provider": "openai",
            "model": "gpt-4",
            "timestamp": "2026-02-20T10:00:00Z",
            "cost_usd": 0.03,
            "tokens_used": 1200,
            "enrichment_types": ["summary", "tags", "highlights"],
        },
        "summary": {
            "short": "REST API design principles.",
            "medium": "A guide to designing RESTful APIs with best practices.",
            "long": "A comprehensive guide to REST API design covering endpoints, status codes, authentication, and versioning.",
        },
        "tags": ["api", "rest", "backend", "http"],
        "topics": ["api", "rest", "backend"],
        "highlights": [
            {"text": "Use nouns for endpoints, not verbs"},
            {"text": "Always version your API"},
        ],
        "chapters": [],
    }

    # Source 2: enriched JSON
    source2 = {
        "enrichment_version": "v1",
        "metadata": {
            "title": "Database Optimization",
            "source_file": "db-optimization.json",
            "provider": "openai",
            "model": "gpt-4",
            "timestamp": "2026-02-20T11:00:00Z",
            "cost_usd": 0.02,
            "tokens_used": 900,
            "enrichment_types": ["summary", "tags"],
        },
        "summary": {
            "short": "Database optimization techniques.",
            "medium": "Techniques for optimizing database queries and schema design.",
        },
        "tags": ["database", "sql", "performance", "backend"],
        "topics": ["database", "optimization"],
        "highlights": [],
        "chapters": [],
    }

    (tmp_path / "rest-api-enriched.json").write_text(
        json.dumps(source1), encoding="utf-8"
    )
    (tmp_path / "db-optimization-enriched.json").write_text(
        json.dumps(source2), encoding="utf-8"
    )

    # Source 3: plain text supplement
    (tmp_path / "notes.txt").write_text(
        "Additional notes on caching strategies and CDN usage.",
        encoding="utf-8",
    )

    return tmp_path


class TestFormatEnhancementsE2E:
    """End-to-end tests for v0.8.7 format enhancements."""

    def test_multi_source_to_blog(self, enriched_source_folder):
        """Full flow: multiple source files → combined → blog output."""
        composer = FormatComposer(llm_enhancer=None)

        result = composer.format_from_sources(
            sources_folder=enriched_source_folder,
            output_type="blog",
            llm_enhance=False,
        )

        assert result.success is True
        assert len(result.content) > 0
        # Blog should contain content from both sources
        assert "API" in result.content or "api" in result.content.lower()

    def test_multi_source_with_image_prompts(self, enriched_source_folder):
        """Full flow: sources → format → image prompts."""
        composer = FormatComposer(llm_enhancer=None)

        # First format from sources
        result = composer.format_from_sources(
            sources_folder=enriched_source_folder,
            output_type="blog",
            llm_enhance=False,
        )
        assert result.success is True

        # Then generate image prompts from the combined content
        combiner = SourceCombiner()
        sources = combiner.load_sources(enriched_source_folder)
        combined = combiner.combine(sources)

        prompts_result = composer.generate_image_prompts(
            enriched_content=combined.enriched_content,
            output_type="blog",
        )
        assert prompts_result is not None
        assert len(prompts_result.prompts) > 0

    def test_multi_source_with_code_samples(self, enriched_source_folder):
        """Full flow: technical sources → format → code samples."""
        composer = FormatComposer(llm_enhancer=None)

        combiner = SourceCombiner()
        sources = combiner.load_sources(enriched_source_folder)
        combined = combiner.combine(sources)

        samples_result = composer.generate_code_samples(
            enriched_content=combined.enriched_content,
            output_type="blog",
        )
        # Content has technical topics (api, backend, database)
        assert samples_result is not None
        assert len(samples_result.samples) > 0

    def test_ai_video_script_output(self, enriched_source_folder):
        """Full flow: sources → ai-video-script output type."""
        composer = FormatComposer(llm_enhancer=None)

        result = composer.format_from_sources(
            sources_folder=enriched_source_folder,
            output_type="ai-video-script",
            platform="youtube",
            llm_enhance=False,
        )
        assert result.success is True
        assert len(result.content) > 0
        # Should contain scene markers
        assert "Scene" in result.content or "scene" in result.content.lower()
