"""
Unit tests for AIVideoScriptGenerator.

Tests scene generation, voiceover timing, platform configurations,
and template rendering for AI video generation tools.
"""

import pytest

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.ai_video_script import AIVideoScriptGenerator
from pipeline.formatters.generators.base_generator import GeneratorConfig
from pipeline.formatters.template_engine import TemplateEngine
from pipeline.formatters.validator import PlatformValidator
from pipeline.formatters.schemas.video_script import PLATFORM_CONFIGS, WORDS_PER_MINUTE


@pytest.fixture
def generator():
    """Create an AIVideoScriptGenerator instance."""
    config = GeneratorConfig(
        template_engine=TemplateEngine(),
        platform_validator=PlatformValidator(),
    )
    return AIVideoScriptGenerator(config)


@pytest.fixture
def enriched_content():
    """Sample enriched content for testing."""
    return {
        "metadata": {
            "title": "Introduction to Machine Learning",
            "source_file": "ml-intro.json",
        },
        "summary": {
            "short": "A beginner's guide to machine learning concepts.",
            "medium": (
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data. This guide covers "
                "supervised learning, unsupervised learning, and neural networks."
            ),
            "long": (
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn and improve from experience "
                "without being explicitly programmed. This comprehensive guide "
                "covers the fundamentals of supervised learning, unsupervised "
                "learning, reinforcement learning, and neural networks. "
                "We explore practical applications in image recognition, "
                "natural language processing, and recommendation systems."
            ),
        },
        "chapters": [
            {"title": "What is Machine Learning?", "summary": "ML is a branch of AI that learns from data."},
            {"title": "Supervised Learning", "summary": "Training models with labeled data for predictions."},
            {"title": "Neural Networks", "summary": "Deep learning architectures inspired by the brain."},
        ],
        "highlights": [
            {"text": "ML models improve with more data"},
            {"text": "Neural networks power modern AI"},
        ],
        "topics": ["machine learning", "artificial intelligence", "deep learning"],
    }


@pytest.fixture
def minimal_content():
    """Minimal enriched content with only required fields."""
    return {
        "summary": {
            "short": "A quick overview.",
        },
    }


class TestAIVideoScriptGeneratorProperties:
    """Tests for generator properties."""

    def test_output_type(self, generator):
        """Test output type is ai-video-script."""
        assert generator.output_type == "ai-video-script"

    def test_supported_platforms(self, generator):
        """Test supported platforms include YouTube, TikTok, Vimeo."""
        platforms = generator.supported_platforms
        assert "youtube" in platforms
        assert "tiktok" in platforms
        assert "vimeo" in platforms

    def test_required_enrichments(self, generator):
        """Test summary is required."""
        assert "summary" in generator.required_enrichments


class TestPlatformConfig:
    """Tests for platform configuration retrieval."""

    def test_get_youtube_config(self, generator):
        """Test YouTube config retrieval."""
        config = generator._get_platform_config("youtube")
        assert config["aspect_ratio"] == "16:9"
        assert config["duration_range"] == (180, 600)

    def test_get_tiktok_config(self, generator):
        """Test TikTok config retrieval."""
        config = generator._get_platform_config("tiktok")
        assert config["aspect_ratio"] == "9:16"
        assert config["duration_range"] == (15, 60)

    def test_get_vimeo_config(self, generator):
        """Test Vimeo config retrieval."""
        config = generator._get_platform_config("vimeo")
        assert config["aspect_ratio"] == "16:9"
        assert config["duration_range"] == (60, 1800)

    def test_default_to_youtube(self, generator):
        """Test unknown platform defaults to YouTube."""
        config = generator._get_platform_config(None)
        assert config == PLATFORM_CONFIGS["youtube"]

    def test_unknown_platform_defaults_to_youtube(self, generator):
        """Test unknown platform name defaults to YouTube."""
        config = generator._get_platform_config("unknown_platform")
        assert config == PLATFORM_CONFIGS["youtube"]


class TestSceneCount:
    """Tests for scene count calculation."""

    def test_short_content_fewer_scenes(self, generator):
        """Test short content produces fewer scenes."""
        config = PLATFORM_CONFIGS["youtube"]
        count = generator._calculate_scene_count(100, config)
        # Short content: max 2 content + 2 (intro/outro) = 4
        assert count <= 4

    def test_medium_content_moderate_scenes(self, generator):
        """Test medium content produces moderate scenes."""
        config = PLATFORM_CONFIGS["youtube"]
        count = generator._calculate_scene_count(1500, config)
        assert count >= 3  # At least intro + 1 content + outro

    def test_long_content_more_scenes(self, generator):
        """Test long content produces more scenes."""
        config = PLATFORM_CONFIGS["youtube"]
        count = generator._calculate_scene_count(5000, config)
        assert count >= 4

    def test_tiktok_fewer_scenes(self, generator):
        """Test TikTok produces fewer scenes due to short duration."""
        config = PLATFORM_CONFIGS["tiktok"]
        count = generator._calculate_scene_count(2000, config)
        # TikTok is short-form, should have fewer scenes
        assert count >= 3  # At least intro + 1 + outro

    def test_always_includes_intro_outro(self, generator):
        """Test scene count always includes intro and outro."""
        for platform in PLATFORM_CONFIGS:
            config = PLATFORM_CONFIGS[platform]
            count = generator._calculate_scene_count(100, config)
            assert count >= 3  # intro + at least 1 content + outro


class TestVoiceoverDuration:
    """Tests for voiceover duration calculation."""

    def test_empty_text_minimum_duration(self, generator):
        """Test empty text returns minimum duration."""
        assert generator._calculate_voiceover_duration("") == 5

    def test_short_text_minimum_duration(self, generator):
        """Test very short text returns minimum duration."""
        assert generator._calculate_voiceover_duration("Hello") == 5

    def test_duration_based_on_word_count(self, generator):
        """Test duration scales with word count."""
        # 150 words at 150 WPM = 60 seconds
        text = " ".join(["word"] * 150)
        duration = generator._calculate_voiceover_duration(text)
        assert duration == 60

    def test_longer_text_longer_duration(self, generator):
        """Test more words produce longer duration."""
        short_text = " ".join(["word"] * 30)
        long_text = " ".join(["word"] * 300)
        short_dur = generator._calculate_voiceover_duration(short_text)
        long_dur = generator._calculate_voiceover_duration(long_text)
        assert long_dur > short_dur

    def test_words_per_minute_constant(self):
        """Test WORDS_PER_MINUTE is 150."""
        assert WORDS_PER_MINUTE == 150


class TestBuildScenes:
    """Tests for scene building logic."""

    def test_scenes_have_intro(self, generator, enriched_content):
        """Test first scene is intro type."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        assert scenes[0]["type"] == "intro"

    def test_scenes_have_outro(self, generator, enriched_content):
        """Test last scene is outro type."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        assert scenes[-1]["type"] == "outro"

    def test_scenes_from_chapters(self, generator, enriched_content):
        """Test content scenes are built from chapters."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        content_scenes = [s for s in scenes if s["type"] == "content"]
        assert len(content_scenes) >= 1

    def test_scenes_without_chapters(self, generator, minimal_content):
        """Test scenes are built from summary when no chapters."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(minimal_content, config, 4)
        assert scenes[0]["type"] == "intro"
        assert scenes[-1]["type"] == "outro"

    def test_scene_numbers_sequential(self, generator, enriched_content):
        """Test scene numbers are sequential starting from 1."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        for i, scene in enumerate(scenes):
            assert scene["scene_number"] == i + 1

    def test_all_scenes_have_required_fields(self, generator, enriched_content):
        """Test every scene has all required fields."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        required_fields = {
            "scene_number", "type", "visual_prompt",
            "voiceover_text", "duration_seconds", "music_suggestion",
        }
        for scene in scenes:
            assert required_fields.issubset(scene.keys()), (
                f"Scene {scene['scene_number']} missing fields"
            )

    def test_music_suggestions_have_required_fields(self, generator, enriched_content):
        """Test music suggestions have mood, genre, tempo."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        for scene in scenes:
            music = scene["music_suggestion"]
            assert "mood" in music
            assert "genre" in music
            assert "tempo" in music

    def test_scene_durations_positive(self, generator, enriched_content):
        """Test all scene durations are positive."""
        config = PLATFORM_CONFIGS["youtube"]
        scenes = generator._build_scenes(enriched_content, config, 5)
        for scene in scenes:
            assert scene["duration_seconds"] > 0


class TestBuildTemplateContext:
    """Tests for template context building."""

    def test_context_has_title(self, generator, enriched_content):
        """Test context includes title."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        context = generator._build_template_context(enriched_content, request)
        assert context["title"] == "Introduction to Machine Learning"

    def test_context_has_platform(self, generator, enriched_content):
        """Test context includes target platform."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="tiktok",
        )
        context = generator._build_template_context(enriched_content, request)
        assert context["target_platform"] == "tiktok"

    def test_context_has_scenes(self, generator, enriched_content):
        """Test context includes scenes list."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        context = generator._build_template_context(enriched_content, request)
        assert len(context["scenes"]) >= 3

    def test_context_has_total_duration(self, generator, enriched_content):
        """Test context includes total duration."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        context = generator._build_template_context(enriched_content, request)
        assert context["total_duration_seconds"] > 0

    def test_context_has_aspect_ratio(self, generator, enriched_content):
        """Test context includes aspect ratio from platform."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="tiktok",
        )
        context = generator._build_template_context(enriched_content, request)
        assert context["aspect_ratio"] == "9:16"

    def test_context_default_platform_youtube(self, generator, enriched_content):
        """Test context defaults to YouTube when no platform specified."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
        )
        context = generator._build_template_context(enriched_content, request)
        assert context["target_platform"] == "youtube"
        assert context["aspect_ratio"] == "16:9"

    def test_context_includes_topics(self, generator, enriched_content):
        """Test context includes topics from enriched content."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        context = generator._build_template_context(enriched_content, request)
        assert "machine learning" in context["topics"]


class TestFormatOutput:
    """Tests for full format() pipeline."""

    def test_format_success(self, generator, enriched_content):
        """Test successful format produces content."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert result.success is True
        assert len(result.content) > 0

    def test_format_contains_title(self, generator, enriched_content):
        """Test formatted output contains the title."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert "Introduction to Machine Learning" in result.content

    def test_format_contains_scene_markers(self, generator, enriched_content):
        """Test formatted output contains scene markers."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert "SCENE 1" in result.content
        assert "INTRO" in result.content

    def test_format_contains_platform_info(self, generator, enriched_content):
        """Test formatted output contains platform information."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="tiktok",
        )
        result = generator.format(request)
        assert "Tiktok" in result.content or "tiktok" in result.content.lower()

    def test_format_contains_production_summary(self, generator, enriched_content):
        """Test formatted output contains production summary."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert "PRODUCTION SUMMARY" in result.content

    def test_format_missing_summary_fails(self, generator):
        """Test format fails when summary is missing."""
        request = FormatRequest(
            enriched_content={"metadata": {"title": "Test"}},
            output_type="ai-video-script",
        )
        result = generator.format(request)
        assert result.success is False
        assert "summary" in result.error.lower()

    def test_format_minimal_content(self, generator, minimal_content):
        """Test format works with minimal content."""
        request = FormatRequest(
            enriched_content=minimal_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert result.success is True

    def test_format_metadata_output_type(self, generator, enriched_content):
        """Test metadata has correct output type."""
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="ai-video-script",
            platform="youtube",
        )
        result = generator.format(request)
        assert result.metadata.output_type == "ai-video-script"


class TestFactoryRegistration:
    """Tests for generator factory registration."""

    def test_registered_in_factory(self):
        """Test AIVideoScriptGenerator is registered in factory."""
        from pipeline.formatters.generators.factory import GeneratorFactory
        assert GeneratorFactory.is_registered("ai-video-script")

    def test_factory_creates_instance(self):
        """Test factory can create AIVideoScriptGenerator instance."""
        from pipeline.formatters.generators.factory import GeneratorFactory
        factory = GeneratorFactory()
        gen = factory.get_generator("ai-video-script")
        assert isinstance(gen, AIVideoScriptGenerator)
        assert gen.output_type == "ai-video-script"
