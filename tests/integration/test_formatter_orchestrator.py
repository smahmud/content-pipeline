"""
Integration tests for FormatComposer orchestrator.

These tests verify the end-to-end functionality of the formatter system:
- Single format generation with LLM enhancement
- Bundle generation with named bundles
- Batch processing with multiple files
- --list-bundles flag
- Error handling for invalid bundle names

Checkpoint Task 15 verification tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.formatters.base import FormatRequest, FormatResult
from pipeline.formatters.bundles.loader import BundleLoader, BundleNotFoundError
from pipeline.formatters.composer import FormatComposer, BundleResult, BatchResult
from pipeline.formatters.llm.enhancer import LLMEnhancer, EnhancementResult


# Sample enriched content for testing - matches EnrichmentV1 schema
SAMPLE_ENRICHED_CONTENT = {
    "enrichment_version": "v1",
    "source_file": "test_video.mp4",
    "metadata": {
        "provider": "bedrock",
        "model": "claude-3-sonnet",
        "timestamp": "2024-01-15T10:00:00Z",
        "cost_usd": 0.05,
        "tokens_used": 1500,
        "enrichment_types": ["summary", "tags", "chapters", "highlights"],
        "title": "Test Video Title",
        "duration_seconds": 600,
    },
    # Enrichments at top level (not nested under "enrichments")
    "summary": {
        "brief": "A brief summary of the test video content.",
        "detailed": "A more detailed summary explaining the key points covered in the video.",
        "key_points": [
            "First key point about the topic",
            "Second key point with more details",
            "Third key point wrapping up",
        ],
    },
    "tags": {
        "primary": ["technology", "tutorial"],
        "secondary": ["python", "programming"],
    },
    "chapters": [
        {"timestamp": "00:00", "title": "Introduction", "summary": "Opening remarks"},
        {"timestamp": "02:30", "title": "Main Content", "summary": "Core discussion"},
        {"timestamp": "08:00", "title": "Conclusion", "summary": "Wrap up"},
    ],
    "highlights": [
        {"text": "This is a key quote from the video.", "timestamp": "01:30"},
        {"text": "Another important highlight.", "timestamp": "05:00"},
    ],
    "transcript": {
        "segments": [
            {"speaker": "Host", "text": "Welcome to the show.", "start": 0, "end": 5},
            {"speaker": "Guest", "text": "Thanks for having me.", "start": 5, "end": 10},
        ],
    },
}


class TestSingleFormatGeneration:
    """Test single format generation (format_single)."""

    def test_format_single_without_llm_enhancement(self):
        """Test single format generation without LLM enhancement."""
        composer = FormatComposer()
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=False,
        )
        
        result = composer.format_single(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata is not None
        assert result.metadata.output_type == "blog"
        # Without LLM enhancement, llm_metadata should be None
        assert result.metadata.llm_metadata is None

    def test_format_single_with_mock_llm_enhancement(self):
        """Test single format generation with mocked LLM enhancement."""
        # Create mock LLM enhancer
        mock_enhancer = MagicMock(spec=LLMEnhancer)
        mock_enhancer.enhance.return_value = EnhancementResult(
            content="Enhanced blog content with better prose.",
            enhanced=True,
            success=True,
            provider="bedrock",
            model="claude-3-sonnet",
            cost_usd=0.001,
            tokens_used=500,
            warnings=[],
        )
        mock_enhancer.estimate_cost.return_value = 0.001
        
        composer = FormatComposer(llm_enhancer=mock_enhancer)
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
            provider="bedrock",
        )
        
        result = composer.format_single(request)
        
        assert result.success is True
        assert result.content == "Enhanced blog content with better prose."
        assert result.metadata is not None
        assert result.metadata.llm_metadata is not None
        assert result.metadata.llm_metadata.enhanced is True
        assert result.metadata.llm_metadata.provider == "bedrock"

    def test_format_single_all_output_types(self):
        """Test that all 16 output types can be generated."""
        composer = FormatComposer()
        
        output_types = [
            "blog", "tweet", "youtube", "seo",
            "linkedin", "newsletter", "chapters", "transcript-clean",
            "podcast-notes", "meeting-minutes", "slides", "notion",
            "obsidian", "quote-cards", "video-script", "tiktok-script",
        ]
        
        for output_type in output_types:
            request = FormatRequest(
                enriched_content=SAMPLE_ENRICHED_CONTENT,
                output_type=output_type,
                llm_enhance=False,
            )
            
            result = composer.format_single(request)
            
            assert result.success is True, f"Failed for output type: {output_type}"
            assert result.content is not None, f"No content for: {output_type}"
            assert len(result.content) > 0, f"Empty content for: {output_type}"

    def test_format_single_with_platform_validation(self):
        """Test single format with platform validation."""
        composer = FormatComposer()
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="tweet",
            platform="twitter",
            llm_enhance=False,
        )
        
        result = composer.format_single(request)
        
        assert result.success is True
        assert result.metadata.validation is not None
        assert result.metadata.validation.platform == "twitter"
        # Twitter limit is 280 characters
        assert result.metadata.validation.character_count <= 280

    def test_format_single_dry_run(self):
        """Test dry run mode returns estimate without execution."""
        mock_enhancer = MagicMock(spec=LLMEnhancer)
        mock_enhancer.estimate_cost.return_value = 0.005
        
        composer = FormatComposer(llm_enhancer=mock_enhancer)
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
            dry_run=True,
        )
        
        result = composer.format_single(request)
        
        assert result.success is True
        assert "Dry run" in result.warnings[0]
        # LLM enhance should NOT have been called
        mock_enhancer.enhance.assert_not_called()


class TestBundleGeneration:
    """Test bundle generation (format_bundle)."""

    def test_format_bundle_blog_launch(self):
        """Test blog-launch bundle generates all expected outputs."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name="blog-launch",
                enriched_content=SAMPLE_ENRICHED_CONTENT,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            assert isinstance(result, BundleResult)
            assert result.bundle_name == "blog-launch"
            # blog-launch should have: blog, tweet, linkedin, seo
            assert "blog" in result.successful
            assert "tweet" in result.successful
            assert "linkedin" in result.successful
            assert "seo" in result.successful
            assert len(result.failed) == 0
            # Manifest should be generated
            assert result.manifest_path != ""
            assert Path(result.manifest_path).exists()

    def test_format_bundle_video_launch(self):
        """Test video-launch bundle generates all expected outputs."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name="video-launch",
                enriched_content=SAMPLE_ENRICHED_CONTENT,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            assert result.bundle_name == "video-launch"
            # video-launch should have: youtube, chapters, tweet, blog
            assert "youtube" in result.successful
            assert "chapters" in result.successful
            assert "tweet" in result.successful
            assert "blog" in result.successful

    def test_format_bundle_all_builtin_bundles(self):
        """Test all 6 built-in bundles can be generated."""
        composer = FormatComposer()
        
        bundle_names = [
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ]
        
        for bundle_name in bundle_names:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = composer.format_bundle(
                    bundle_name=bundle_name,
                    enriched_content=SAMPLE_ENRICHED_CONTENT,
                    output_dir=tmpdir,
                    llm_enhance=False,
                )
                
                assert result.bundle_name == bundle_name
                assert len(result.successful) > 0, f"No outputs for bundle: {bundle_name}"
                # Manifest should exist
                assert Path(result.manifest_path).exists()

    def test_format_bundle_error_isolation(self):
        """Test that bundle continues if one output type fails."""
        # Create a composer with a generator that fails for one type
        composer = FormatComposer()
        
        # Patch one generator to fail
        original_format_single = composer.format_single
        
        def mock_format_single(request):
            if request.output_type == "seo":
                raise Exception("Simulated SEO failure")
            return original_format_single(request)
        
        with patch.object(composer, 'format_single', side_effect=mock_format_single):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = composer.format_bundle(
                    bundle_name="blog-launch",
                    enriched_content=SAMPLE_ENRICHED_CONTENT,
                    output_dir=tmpdir,
                    llm_enhance=False,
                )
                
                # Should have some successful outputs
                assert len(result.successful) > 0
                # SEO should be in failed
                assert any("seo" in f[0] for f in result.failed)

    def test_format_bundle_manifest_content(self):
        """Test manifest file contains expected fields."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name="social-only",
                enriched_content=SAMPLE_ENRICHED_CONTENT,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Read manifest
            with open(result.manifest_path, "r") as f:
                manifest = json.load(f)
            
            assert manifest["manifest_version"] == "v1"
            assert manifest["bundle_name"] == "social-only"
            assert "outputs_requested" in manifest
            assert "outputs_successful" in manifest
            assert "timestamp" in manifest


class TestBundleListing:
    """Test --list-bundles functionality."""

    def test_list_bundles_returns_all_builtin(self):
        """Test list_bundles returns all 6 built-in bundles."""
        composer = FormatComposer()
        
        bundles = composer.list_bundles()
        
        assert len(bundles) >= 6
        bundle_names = [b.name for b in bundles]
        assert "blog-launch" in bundle_names
        assert "video-launch" in bundle_names
        assert "podcast" in bundle_names
        assert "social-only" in bundle_names
        assert "full-repurpose" in bundle_names
        assert "notes-package" in bundle_names

    def test_get_bundle_names(self):
        """Test get_bundle_names returns sorted list."""
        composer = FormatComposer()
        
        names = composer.get_bundle_names()
        
        assert isinstance(names, list)
        assert len(names) >= 6
        # Should be sorted
        assert names == sorted(names)

    def test_format_bundle_list_for_cli(self):
        """Test format_bundle_list produces CLI-friendly output."""
        composer = FormatComposer()
        
        output = composer.format_bundle_list()
        
        assert "Available bundles:" in output
        assert "blog-launch" in output
        assert "video-launch" in output
        # Should include descriptions
        assert "Blog article with social promotion" in output
        # Should include outputs
        assert "Outputs:" in output

    def test_has_bundle(self):
        """Test has_bundle correctly identifies existing bundles."""
        composer = FormatComposer()
        
        assert composer.has_bundle("blog-launch") is True
        assert composer.has_bundle("video-launch") is True
        assert composer.has_bundle("nonexistent-bundle") is False


class TestInvalidBundleErrorHandling:
    """Test error handling for invalid bundle names."""

    def test_invalid_bundle_raises_bundle_not_found_error(self):
        """Test that invalid bundle name raises BundleNotFoundError."""
        composer = FormatComposer()
        
        with pytest.raises(BundleNotFoundError) as exc_info:
            composer.format_bundle(
                bundle_name="nonexistent-bundle",
                enriched_content=SAMPLE_ENRICHED_CONTENT,
            )
        
        error = exc_info.value
        assert error.bundle_name == "nonexistent-bundle"
        assert len(error.available_bundles) >= 6

    def test_invalid_bundle_error_message_lists_available(self):
        """Test error message includes list of available bundles."""
        composer = FormatComposer()
        
        with pytest.raises(BundleNotFoundError) as exc_info:
            composer.format_bundle(
                bundle_name="my-custom-bundle",
                enriched_content=SAMPLE_ENRICHED_CONTENT,
            )
        
        error_message = str(exc_info.value)
        assert "my-custom-bundle" in error_message
        assert "not found" in error_message.lower()
        assert "Available bundles:" in error_message
        # Should list some available bundles
        assert "blog-launch" in error_message

    def test_bundle_loader_raises_for_invalid_name(self):
        """Test BundleLoader directly raises appropriate error."""
        loader = BundleLoader()
        
        with pytest.raises(BundleNotFoundError) as exc_info:
            loader.load_bundle("invalid-name")
        
        assert exc_info.value.bundle_name == "invalid-name"


class TestBatchProcessing:
    """Test batch processing (format_batch)."""

    def test_batch_processing_multiple_files(self):
        """Test batch processing with multiple input files."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple enriched files
            for i in range(3):
                content = SAMPLE_ENRICHED_CONTENT.copy()
                content["source_file"] = f"video_{i}.mp4"
                
                file_path = Path(tmpdir) / f"video_{i}.enriched.json"
                with open(file_path, "w") as f:
                    json.dump(content, f)
            
            # Run batch processing
            result = composer.format_batch(
                input_pattern=str(Path(tmpdir) / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            assert isinstance(result, BatchResult)
            assert len(result.successful) == 3
            assert len(result.failed) == 0
            assert result.total_time > 0

    def test_batch_processing_error_isolation(self):
        """Test batch continues processing after file failure."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid files
            for i in range(2):
                content = SAMPLE_ENRICHED_CONTENT.copy()
                content["source_file"] = f"video_{i}.mp4"
                
                file_path = Path(tmpdir) / f"video_{i}.enriched.json"
                with open(file_path, "w") as f:
                    json.dump(content, f)
            
            # Create invalid file
            invalid_path = Path(tmpdir) / "invalid.enriched.json"
            with open(invalid_path, "w") as f:
                f.write("not valid json {{{")
            
            # Run batch processing
            result = composer.format_batch(
                input_pattern=str(Path(tmpdir) / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Should have processed valid files
            assert len(result.successful) == 2
            # Should have recorded failure
            assert len(result.failed) == 1
            assert "invalid.enriched.json" in result.failed[0][0]

    def test_batch_processing_empty_pattern(self):
        """Test batch with no matching files returns empty result."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_batch(
                input_pattern=str(Path(tmpdir) / "*.nonexistent"),
                output_type="blog",
                output_dir=tmpdir,
            )
            
            assert len(result.successful) == 0
            assert len(result.failed) == 0

    def test_batch_summary_format(self):
        """Test batch summary formatting for CLI."""
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one file
            content = SAMPLE_ENRICHED_CONTENT.copy()
            file_path = Path(tmpdir) / "test.enriched.json"
            with open(file_path, "w") as f:
                json.dump(content, f)
            
            result = composer.format_batch(
                input_pattern=str(Path(tmpdir) / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            summary = composer.format_batch_summary(result)
            
            assert "Batch Processing Summary:" in summary
            assert "Total files:" in summary
            assert "Successful:" in summary
            assert "Total time:" in summary


class TestCostControl:
    """Test cost estimation and control."""

    def test_cost_estimation_without_llm(self):
        """Test cost estimation returns zero without LLM."""
        composer = FormatComposer()  # No LLM enhancer
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
        )
        
        estimate = composer.estimate_cost(request)
        
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.within_budget is True

    def test_cost_estimation_with_mock_llm(self):
        """Test cost estimation with mocked LLM."""
        mock_enhancer = MagicMock(spec=LLMEnhancer)
        mock_enhancer.estimate_cost.return_value = 0.01
        
        composer = FormatComposer(llm_enhancer=mock_enhancer)
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
            max_cost=0.05,
        )
        
        estimate = composer.estimate_cost(request)
        
        assert estimate.estimated_cost_usd == 0.01
        assert estimate.within_budget is True

    def test_cost_limit_exceeded_raises_error(self):
        """Test that exceeding cost limit raises error."""
        mock_enhancer = MagicMock(spec=LLMEnhancer)
        mock_enhancer.estimate_cost.return_value = 0.10  # High cost
        
        composer = FormatComposer(llm_enhancer=mock_enhancer)
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
            max_cost=0.01,  # Low limit
        )
        
        from pipeline.formatters.errors import CostLimitExceededError
        
        with pytest.raises(CostLimitExceededError):
            composer.format_single(request)


class TestGracefulDegradation:
    """Test graceful degradation when LLM fails."""

    def test_llm_failure_returns_template_output(self):
        """Test that LLM failure falls back to template-only output."""
        mock_enhancer = MagicMock(spec=LLMEnhancer)
        mock_enhancer.enhance.return_value = EnhancementResult(
            content="Original template content",
            enhanced=False,
            success=False,
            provider="bedrock",
            model="claude-3-sonnet",
            cost_usd=0.0,
            tokens_used=0,
            warnings=["LLM enhancement failed, using template output"],
        )
        mock_enhancer.estimate_cost.return_value = 0.001
        
        composer = FormatComposer(llm_enhancer=mock_enhancer)
        
        request = FormatRequest(
            enriched_content=SAMPLE_ENRICHED_CONTENT,
            output_type="blog",
            llm_enhance=True,
        )
        
        result = composer.format_single(request)
        
        # Should still succeed with template output
        assert result.success is True
        assert result.content is not None
        # Should have warning about fallback
        assert any("failed" in w.lower() or "template" in w.lower() for w in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
