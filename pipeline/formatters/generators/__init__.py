"""
Output type generators implementing BaseFormatter protocol.

Each generator is responsible for transforming enriched content
into a specific output format (blog, tweet, youtube, etc.).

All 16 output types are supported:
- Tier 1: blog, tweet, youtube, seo
- Tier 2: linkedin, newsletter, chapters, transcript-clean
- Tier 3: podcast-notes, meeting-minutes, slides, notion, obsidian, quote-cards
- AI Video: video-script, tiktok-script
"""

from pipeline.formatters.generators.base_generator import (
    BaseGenerator,
    GeneratorConfig,
)
from pipeline.formatters.generators.factory import (
    GeneratorFactory,
    GeneratorFactoryError,
    register_all_generators,
)

# Import all generators to trigger registration
from pipeline.formatters.generators.blog import BlogGenerator
from pipeline.formatters.generators.tweet import TweetGenerator
from pipeline.formatters.generators.youtube import YouTubeGenerator
from pipeline.formatters.generators.seo import SEOGenerator
from pipeline.formatters.generators.linkedin import LinkedInGenerator
from pipeline.formatters.generators.newsletter import NewsletterGenerator
from pipeline.formatters.generators.chapters import ChaptersGenerator
from pipeline.formatters.generators.transcript_clean import TranscriptCleanGenerator
from pipeline.formatters.generators.podcast_notes import PodcastNotesGenerator
from pipeline.formatters.generators.meeting_minutes import MeetingMinutesGenerator
from pipeline.formatters.generators.slides import SlidesGenerator
from pipeline.formatters.generators.notion import NotionGenerator
from pipeline.formatters.generators.obsidian import ObsidianGenerator
from pipeline.formatters.generators.quote_cards import QuoteCardsGenerator
from pipeline.formatters.generators.video_script import VideoScriptGenerator
from pipeline.formatters.generators.tiktok_script import TikTokScriptGenerator


__all__ = [
    # Base classes
    "BaseGenerator",
    "GeneratorConfig",
    "GeneratorFactory",
    "GeneratorFactoryError",
    "register_all_generators",
    # Tier 1 generators
    "BlogGenerator",
    "TweetGenerator",
    "YouTubeGenerator",
    "SEOGenerator",
    # Tier 2 generators
    "LinkedInGenerator",
    "NewsletterGenerator",
    "ChaptersGenerator",
    "TranscriptCleanGenerator",
    # Tier 3 generators
    "PodcastNotesGenerator",
    "MeetingMinutesGenerator",
    "SlidesGenerator",
    "NotionGenerator",
    "ObsidianGenerator",
    "QuoteCardsGenerator",
    # AI Video generators
    "VideoScriptGenerator",
    "TikTokScriptGenerator",
]
