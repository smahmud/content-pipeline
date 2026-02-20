"""
Unit tests for ImagePromptGenerator.

Tests prompt generation for different platforms and output types.
"""

import json
import pytest

from pipeline.formatters.image_prompts import (
    DEFAULT_DIMENSIONS,
    ImagePrompt,
    ImagePromptGenerator,
    ImagePromptsResult,
    PLATFORM_IMAGE_COUNTS,
    SUPPORTED_OUTPUT_TYPES,
    UNSUPPORTED_OUTPUT_TYPES,
)


class TestImagePromptDataclass:
    """Tests for ImagePrompt dataclass."""
    
    def test_create_image_prompt(self):
        """Test basic ImagePrompt creation."""
        prompt = ImagePrompt(
            prompt="A beautiful sunset over mountains",
            image_type="header",
            suggested_dimensions=(1200, 630),
            position_hint="article header",
            style_notes="Warm colors"
        )
        assert prompt.prompt == "A beautiful sunset over mountains"
        assert prompt.image_type == "header"
        assert prompt.suggested_dimensions == (1200, 630)
        assert prompt.position_hint == "article header"
        assert prompt.style_notes == "Warm colors"
    
    def test_image_prompt_optional_style_notes(self):
        """Test ImagePrompt with default style_notes."""
        prompt = ImagePrompt(
            prompt="Test prompt",
            image_type="diagram",
            suggested_dimensions=(800, 600),
            position_hint="section 1"
        )
        assert prompt.style_notes is None


class TestImagePromptsResultDataclass:
    """Tests for ImagePromptsResult dataclass."""
    
    def test_create_result(self):
        """Test basic ImagePromptsResult creation."""
        result = ImagePromptsResult(
            prompts=[],
            output_type="blog",
            platform="medium",
            content_title="Test Article"
        )
        assert result.prompts == []
        assert result.output_type == "blog"
        assert result.platform == "medium"
        assert result.content_title == "Test Article"
        assert result.schema_version == "image_prompts_v1"
    
    def test_result_with_prompts(self):
        """Test ImagePromptsResult with prompts."""
        prompts = [
            ImagePrompt("Prompt 1", "header", (1200, 630), "header"),
            ImagePrompt("Prompt 2", "diagram", (800, 600), "section 1"),
        ]
        result = ImagePromptsResult(
            prompts=prompts,
            output_type="blog",
            content_title="Test"
        )
        assert len(result.prompts) == 2


class TestImagePromptGeneratorSupport:
    """Tests for output type support checking."""
    
    def test_supported_output_types(self):
        """Test that supported types return True."""
        generator = ImagePromptGenerator()
        
        for output_type in SUPPORTED_OUTPUT_TYPES:
            assert generator.is_supported(output_type), f"{output_type} should be supported"
    
    def test_unsupported_output_types(self):
        """Test that unsupported types return False."""
        generator = ImagePromptGenerator()
        
        for output_type in UNSUPPORTED_OUTPUT_TYPES:
            assert not generator.is_supported(output_type), f"{output_type} should not be supported"
    
    def test_unknown_output_type(self):
        """Test that unknown types return False."""
        generator = ImagePromptGenerator()
        assert not generator.is_supported("unknown-type")


class TestImagePromptGeneratorBlog:
    """Tests for blog/tutorial prompt generation."""
    
    def test_generate_blog_prompts(self):
        """Test generating prompts for blog output."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Python Best Practices"},
            "summary": {"medium": "A guide to Python best practices"},
            "topics": ["python", "programming", "best practices"],
            "tags": ["python", "coding", "tutorial"],
        }
        
        result = generator.generate(enriched_content, "blog")
        
        assert result.output_type == "blog"
        assert result.content_title == "Python Best Practices"
        assert len(result.prompts) > 0
        
        # Should have a header
        header_prompts = [p for p in result.prompts if p.image_type == "header"]
        assert len(header_prompts) >= 1
    
    def test_generate_blog_with_platform(self):
        """Test generating prompts for blog with specific platform."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Test Article"},
            "summary": {"short": "Test summary"},
            "topics": ["testing"],
        }
        
        result = generator.generate(enriched_content, "blog", platform="medium")
        
        assert result.platform == "medium"
        # Medium config: header=1, supporting=2, total_max=3
        assert len(result.prompts) <= 3
    
    def test_generate_tutorial_prompts(self):
        """Test generating prompts for tutorial output."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Docker Tutorial"},
            "summary": {"medium": "Learn Docker basics"},
            "topics": ["docker", "containers", "devops"],
        }
        
        result = generator.generate(enriched_content, "tutorial")
        
        assert result.output_type == "tutorial"
        # Tutorial should have more supporting images
        assert len(result.prompts) >= 3


class TestImagePromptGeneratorLinkedIn:
    """Tests for LinkedIn prompt generation."""
    
    def test_generate_linkedin_prompts(self):
        """Test generating prompts for LinkedIn."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Career Advice"},
            "summary": {"short": "Professional tips"},
            "topics": ["career", "professional development"],
        }
        
        result = generator.generate(enriched_content, "linkedin")
        
        assert result.output_type == "linkedin"
        # LinkedIn should have exactly 1 header image
        assert len(result.prompts) == 1
        assert result.prompts[0].image_type == "header"
        # LinkedIn dimensions
        assert result.prompts[0].suggested_dimensions == (1200, 627)


class TestImagePromptGeneratorYouTube:
    """Tests for YouTube prompt generation."""
    
    def test_generate_youtube_prompts(self):
        """Test generating prompts for YouTube."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Tech Review"},
            "summary": {"short": "Product review"},
            "topics": ["technology", "reviews"],
        }
        
        result = generator.generate(enriched_content, "youtube")
        
        assert result.output_type == "youtube"
        # YouTube should have thumbnail and end screen
        assert len(result.prompts) == 2
        
        image_types = {p.image_type for p in result.prompts}
        assert "thumbnail" in image_types
        assert "end_screen" in image_types


class TestImagePromptGeneratorSlides:
    """Tests for slides prompt generation."""
    
    def test_generate_slides_prompts(self):
        """Test generating prompts for slides."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Quarterly Report"},
            "summary": {"short": "Q4 results"},
            "topics": ["sales", "marketing", "growth"],
        }
        
        result = generator.generate(enriched_content, "slides")
        
        assert result.output_type == "slides"
        # Should have title slide + concept illustrations
        assert len(result.prompts) >= 2


class TestImagePromptGeneratorNewsletter:
    """Tests for newsletter prompt generation."""
    
    def test_generate_newsletter_prompts(self):
        """Test generating prompts for newsletter."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Weekly Digest"},
            "summary": {"short": "This week's updates"},
            "topics": ["news", "updates"],
        }
        
        result = generator.generate(enriched_content, "newsletter")
        
        assert result.output_type == "newsletter"
        # Should have banner + dividers
        assert len(result.prompts) >= 2
        
        image_types = {p.image_type for p in result.prompts}
        assert "banner" in image_types


class TestImagePromptGeneratorUnsupported:
    """Tests for unsupported output types."""
    
    def test_unsupported_type_returns_empty(self):
        """Test that unsupported types return empty prompts."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Test"},
            "summary": {"short": "Test"},
        }
        
        result = generator.generate(enriched_content, "tweet")
        
        assert result.output_type == "tweet"
        assert len(result.prompts) == 0
    
    def test_unsupported_type_preserves_title(self):
        """Test that unsupported types still extract title."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "My Tweet Content"},
        }
        
        result = generator.generate(enriched_content, "tweet")
        
        assert result.content_title == "My Tweet Content"


class TestImagePromptGeneratorJSON:
    """Tests for JSON serialization."""
    
    def test_to_json(self):
        """Test converting result to JSON."""
        generator = ImagePromptGenerator()
        
        prompts = [
            ImagePrompt(
                prompt="Test prompt",
                image_type="header",
                suggested_dimensions=(1200, 630),
                position_hint="header",
                style_notes="Professional"
            )
        ]
        result = ImagePromptsResult(
            prompts=prompts,
            output_type="blog",
            platform="medium",
            content_title="Test Article"
        )
        
        json_str = generator.to_json(result)
        data = json.loads(json_str)
        
        assert data["schema_version"] == "image_prompts_v1"
        assert data["content_title"] == "Test Article"
        assert data["output_type"] == "blog"
        assert data["platform"] == "medium"
        assert data["prompt_count"] == 1
        assert len(data["prompts"]) == 1
        assert data["prompts"][0]["prompt"] == "Test prompt"
        assert data["prompts"][0]["suggested_dimensions"] == [1200, 630]


class TestImagePromptGeneratorFilename:
    """Tests for output filename generation."""
    
    def test_get_output_filename(self):
        """Test generating image prompts filename."""
        generator = ImagePromptGenerator()
        
        filename = generator.get_output_filename("output/my-blog.md")
        # Use Path for cross-platform comparison
        from pathlib import Path
        assert Path(filename) == Path("output/my-blog-image-prompts.json")
    
    def test_get_output_filename_json_input(self):
        """Test filename generation with JSON input."""
        generator = ImagePromptGenerator()
        
        filename = generator.get_output_filename("content.json")
        assert filename == "content-image-prompts.json"
    
    def test_get_output_filename_nested_path(self):
        """Test filename generation with nested path."""
        generator = ImagePromptGenerator()
        
        filename = generator.get_output_filename("path/to/output/article.md")
        from pathlib import Path
        assert Path(filename) == Path("path/to/output/article-image-prompts.json")


class TestPlatformConfigs:
    """Tests for platform configuration constants."""
    
    def test_all_platforms_have_required_keys(self):
        """Test all platform configs have required keys."""
        for platform, config in PLATFORM_IMAGE_COUNTS.items():
            assert "total_max" in config, f"{platform} missing total_max"
    
    def test_medium_config(self):
        """Test Medium platform configuration."""
        config = PLATFORM_IMAGE_COUNTS["medium"]
        assert config["header"] == 1
        assert config["supporting"] == 2
        assert config["total_max"] == 3
    
    def test_youtube_config(self):
        """Test YouTube platform configuration."""
        config = PLATFORM_IMAGE_COUNTS["youtube"]
        assert config["thumbnail"] == 1
        assert config["end_screen"] == 1


class TestDefaultDimensions:
    """Tests for default dimension constants."""
    
    def test_header_dimensions(self):
        """Test header image dimensions."""
        assert DEFAULT_DIMENSIONS["header"] == (1200, 630)
    
    def test_thumbnail_dimensions(self):
        """Test thumbnail dimensions."""
        assert DEFAULT_DIMENSIONS["thumbnail"] == (1280, 720)
    
    def test_all_dimensions_are_tuples(self):
        """Test all dimensions are valid tuples."""
        for image_type, dims in DEFAULT_DIMENSIONS.items():
            assert isinstance(dims, tuple), f"{image_type} dimensions not a tuple"
            assert len(dims) == 2, f"{image_type} dimensions should have 2 values"
            assert dims[0] > 0 and dims[1] > 0, f"{image_type} dimensions should be positive"


class TestImagePromptSchemaCompleteness:
    """Tests for prompt schema completeness (Property 4)."""
    
    def test_all_prompts_have_required_fields(self):
        """Test that all generated prompts have required fields."""
        generator = ImagePromptGenerator()
        
        enriched_content = {
            "metadata": {"title": "Test"},
            "summary": {"medium": "Test summary"},
            "topics": ["topic1", "topic2"],
        }
        
        # Test all supported output types
        for output_type in SUPPORTED_OUTPUT_TYPES:
            result = generator.generate(enriched_content, output_type)
            
            for prompt in result.prompts:
                # Required fields
                assert prompt.prompt, f"Empty prompt for {output_type}"
                assert len(prompt.prompt) > 0, f"Empty prompt string for {output_type}"
                assert prompt.image_type, f"Missing image_type for {output_type}"
                assert prompt.suggested_dimensions, f"Missing dimensions for {output_type}"
                assert len(prompt.suggested_dimensions) == 2, f"Invalid dimensions for {output_type}"
                assert prompt.suggested_dimensions[0] > 0, f"Invalid width for {output_type}"
                assert prompt.suggested_dimensions[1] > 0, f"Invalid height for {output_type}"
                assert prompt.position_hint, f"Missing position_hint for {output_type}"
