"""
Unit tests for AI Video Script schemas.

Tests dataclass instantiation, serialization, and platform configurations.
"""

import pytest
from dataclasses import asdict

from pipeline.formatters.schemas.video_script import (
    AIVideoScript,
    MusicSuggestion,
    PLATFORM_CONFIGS,
    VideoMetadata,
    VideoScene,
    WORDS_PER_MINUTE,
)


class TestMusicSuggestion:
    """Tests for MusicSuggestion dataclass."""
    
    def test_create_music_suggestion(self):
        """Test basic MusicSuggestion creation."""
        music = MusicSuggestion(
            mood="upbeat",
            genre="electronic",
            tempo="medium"
        )
        assert music.mood == "upbeat"
        assert music.genre == "electronic"
        assert music.tempo == "medium"
    
    def test_music_suggestion_to_dict(self):
        """Test MusicSuggestion serialization to dict."""
        music = MusicSuggestion(
            mood="calm",
            genre="acoustic",
            tempo="slow"
        )
        result = asdict(music)
        assert result == {
            "mood": "calm",
            "genre": "acoustic",
            "tempo": "slow"
        }
    
    def test_music_suggestion_tempo_values(self):
        """Test all valid tempo values."""
        for tempo in ["slow", "medium", "fast"]:
            music = MusicSuggestion(mood="test", genre="test", tempo=tempo)
            assert music.tempo == tempo


class TestVideoScene:
    """Tests for VideoScene dataclass."""
    
    def test_create_video_scene(self):
        """Test basic VideoScene creation."""
        music = MusicSuggestion(mood="inspiring", genre="orchestral", tempo="medium")
        scene = VideoScene(
            scene_number=1,
            duration_seconds=15,
            visual_prompt="A sunrise over mountains",
            voiceover_text="Welcome to our journey.",
            music_suggestion=music
        )
        assert scene.scene_number == 1
        assert scene.duration_seconds == 15
        assert scene.visual_prompt == "A sunrise over mountains"
        assert scene.voiceover_text == "Welcome to our journey."
        assert scene.music_suggestion.mood == "inspiring"
    
    def test_video_scene_to_dict(self):
        """Test VideoScene serialization to dict."""
        music = MusicSuggestion(mood="dramatic", genre="electronic", tempo="fast")
        scene = VideoScene(
            scene_number=2,
            duration_seconds=30,
            visual_prompt="Fast-paced city traffic",
            voiceover_text="The world moves quickly.",
            music_suggestion=music
        )
        result = asdict(scene)
        assert result["scene_number"] == 2
        assert result["duration_seconds"] == 30
        assert result["music_suggestion"]["mood"] == "dramatic"


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""
    
    def test_create_youtube_metadata(self):
        """Test VideoMetadata for YouTube."""
        metadata = VideoMetadata(
            title="Backend Developer Roadmap 2024",
            target_platform="youtube",
            aspect_ratio="16:9",
            recommended_duration_range=(180, 600)
        )
        assert metadata.title == "Backend Developer Roadmap 2024"
        assert metadata.target_platform == "youtube"
        assert metadata.aspect_ratio == "16:9"
        assert metadata.recommended_duration_range == (180, 600)
    
    def test_create_tiktok_metadata(self):
        """Test VideoMetadata for TikTok."""
        metadata = VideoMetadata(
            title="Quick Coding Tip",
            target_platform="tiktok",
            aspect_ratio="9:16",
            recommended_duration_range=(15, 60)
        )
        assert metadata.target_platform == "tiktok"
        assert metadata.aspect_ratio == "9:16"
    
    def test_create_vimeo_metadata(self):
        """Test VideoMetadata for Vimeo."""
        metadata = VideoMetadata(
            title="Documentary Style Video",
            target_platform="vimeo",
            aspect_ratio="16:9",
            recommended_duration_range=(60, 1800)
        )
        assert metadata.target_platform == "vimeo"


class TestAIVideoScript:
    """Tests for AIVideoScript dataclass."""
    
    def test_create_empty_script(self):
        """Test creating an empty AIVideoScript."""
        script = AIVideoScript()
        assert script.schema_version == "ai_video_script_v1"
        assert script.metadata is None
        assert script.scenes == []
        assert script.total_duration_seconds == 0
    
    def test_create_script_with_scenes(self):
        """Test creating AIVideoScript with scenes."""
        music = MusicSuggestion(mood="upbeat", genre="pop", tempo="medium")
        scenes = [
            VideoScene(1, 10, "Intro scene", "Hello!", music),
            VideoScene(2, 20, "Main content", "Let's dive in.", music),
            VideoScene(3, 10, "Outro scene", "Thanks for watching!", music),
        ]
        metadata = VideoMetadata(
            title="Test Video",
            target_platform="youtube",
            aspect_ratio="16:9",
            recommended_duration_range=(180, 600)
        )
        
        script = AIVideoScript(
            metadata=metadata,
            scenes=scenes
        )
        
        assert len(script.scenes) == 3
        assert script.total_duration_seconds == 40  # 10 + 20 + 10
        assert script.metadata.title == "Test Video"
    
    def test_script_auto_calculates_duration(self):
        """Test that total_duration_seconds is auto-calculated from scenes."""
        music = MusicSuggestion(mood="calm", genre="ambient", tempo="slow")
        scenes = [
            VideoScene(1, 15, "Scene 1", "Text 1", music),
            VideoScene(2, 25, "Scene 2", "Text 2", music),
            VideoScene(3, 35, "Scene 3", "Text 3", music),
        ]
        
        script = AIVideoScript(scenes=scenes)
        assert script.total_duration_seconds == 75  # 15 + 25 + 35
    
    def test_script_respects_explicit_duration(self):
        """Test that explicit total_duration_seconds is preserved."""
        music = MusicSuggestion(mood="calm", genre="ambient", tempo="slow")
        scenes = [
            VideoScene(1, 10, "Scene 1", "Text 1", music),
        ]
        
        # Explicit duration should be preserved (not overwritten)
        script = AIVideoScript(scenes=scenes, total_duration_seconds=100)
        assert script.total_duration_seconds == 100


class TestPlatformConfigs:
    """Tests for platform configuration constants."""
    
    def test_youtube_config(self):
        """Test YouTube platform configuration."""
        config = PLATFORM_CONFIGS["youtube"]
        assert config["aspect_ratio"] == "16:9"
        assert config["duration_range"] == (180, 600)
        assert config["scene_duration_range"] == (10, 30)
    
    def test_tiktok_config(self):
        """Test TikTok platform configuration."""
        config = PLATFORM_CONFIGS["tiktok"]
        assert config["aspect_ratio"] == "9:16"
        assert config["duration_range"] == (15, 60)
        assert config["scene_duration_range"] == (3, 10)
    
    def test_vimeo_config(self):
        """Test Vimeo platform configuration."""
        config = PLATFORM_CONFIGS["vimeo"]
        assert config["aspect_ratio"] == "16:9"
        assert config["duration_range"] == (60, 1800)
        assert config["scene_duration_range"] == (15, 45)
    
    def test_all_platforms_have_required_keys(self):
        """Test all platforms have required configuration keys."""
        required_keys = {"aspect_ratio", "duration_range", "scene_duration_range"}
        for platform, config in PLATFORM_CONFIGS.items():
            assert required_keys.issubset(config.keys()), f"{platform} missing keys"


class TestConstants:
    """Tests for module constants."""
    
    def test_words_per_minute(self):
        """Test WORDS_PER_MINUTE constant."""
        assert WORDS_PER_MINUTE == 150
        assert isinstance(WORDS_PER_MINUTE, int)
