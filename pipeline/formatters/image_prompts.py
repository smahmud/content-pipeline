"""
Image prompt generator for AI image generation tools.

Generates detailed text prompts for AI image generators like
DALL-E, Midjourney, Stable Diffusion, and Gemini Imagen.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImagePrompt:
    """A single image prompt for AI generation.
    
    Attributes:
        prompt: Detailed text description for image generation
        image_type: Type of image (header, infographic, diagram, illustration, thumbnail)
        suggested_dimensions: Width and height in pixels
        position_hint: Where the image should appear in content
        style_notes: Optional style guidance
    """
    prompt: str
    image_type: Literal["header", "infographic", "diagram", "illustration", "thumbnail", "end_screen", "banner", "divider"]
    suggested_dimensions: tuple[int, int]  # (width, height)
    position_hint: str
    style_notes: Optional[str] = None


@dataclass
class ImagePromptsResult:
    """Collection of image prompts for content.
    
    Attributes:
        prompts: List of generated image prompts
        output_type: The format type this was generated for
        platform: Target platform if specified
        content_title: Title of the source content
        schema_version: Version of the output schema
    """
    prompts: list[ImagePrompt]
    output_type: str
    platform: Optional[str] = None
    content_title: str = ""
    schema_version: str = "image_prompts_v1"


# Output types that support image prompts
SUPPORTED_OUTPUT_TYPES = {
    "blog", "tutorial", "linkedin", "youtube", 
    "slides", "newsletter"
}

# Output types that do NOT support image prompts
UNSUPPORTED_OUTPUT_TYPES = {
    "tweet", "chapters", "transcript-clean", "meeting-minutes",
    "podcast-notes", "notion", "obsidian", "quote-cards",
    "video-script", "tiktok-script", "ai-video-script", "seo"
}

# Platform-specific image configurations
PLATFORM_IMAGE_COUNTS = {
    # Blog platforms
    "medium": {"header": 1, "supporting": 2, "total_max": 3},
    "wordpress": {"header": 1, "supporting": 4, "total_max": 5},
    "ghost": {"header": 1, "supporting": 3, "total_max": 4},
    "substack": {"header": 1, "supporting": 2, "total_max": 3},
    
    # Default for blog/tutorial without platform
    "blog_default": {"header": 1, "supporting": 4, "total_max": 5},
    "tutorial_default": {"header": 1, "supporting": 5, "total_max": 6},
    
    # Social/professional
    "linkedin": {"header": 1, "supporting": 0, "total_max": 1},
    
    # Video platforms
    "youtube": {"thumbnail": 1, "end_screen": 1, "total_max": 2},
    
    # Presentations
    "slides": {"title": 1, "concepts": 5, "total_max": 6},
    
    # Newsletter
    "newsletter": {"banner": 1, "dividers": 2, "total_max": 3},
}

# Default dimensions by image type
DEFAULT_DIMENSIONS = {
    "header": (1200, 630),      # Open Graph standard
    "thumbnail": (1280, 720),   # YouTube thumbnail
    "end_screen": (1920, 1080), # YouTube end screen
    "infographic": (800, 1200), # Tall infographic
    "diagram": (1000, 800),     # Technical diagram
    "illustration": (800, 600), # General illustration
    "banner": (1200, 400),      # Newsletter banner
    "divider": (800, 200),      # Section divider
}


class ImagePromptGenerator:
    """Generates AI image prompts for visual content.
    
    Analyzes enriched content and generates detailed prompts
    suitable for AI image generation tools.
    """
    
    def __init__(self, llm_enhancer=None):
        """Initialize the image prompt generator.
        
        Args:
            llm_enhancer: Optional LLM enhancer for advanced prompt generation
        """
        self.llm_enhancer = llm_enhancer
    
    def is_supported(self, output_type: str) -> bool:
        """Check if output type supports image prompts.
        
        Args:
            output_type: The format output type
            
        Returns:
            True if image prompts are supported
        """
        return output_type in SUPPORTED_OUTPUT_TYPES
    
    def generate(
        self,
        enriched_content: dict,
        output_type: str,
        platform: Optional[str] = None,
    ) -> ImagePromptsResult:
        """Generate image prompts for content.
        
        Args:
            enriched_content: The enriched content dictionary
            output_type: Target output format type
            platform: Optional target platform
            
        Returns:
            ImagePromptsResult with generated prompts
        """
        if not self.is_supported(output_type):
            logger.warning(f"Image prompts not supported for output type: {output_type}")
            return ImagePromptsResult(
                prompts=[],
                output_type=output_type,
                platform=platform,
                content_title=self._get_title(enriched_content),
            )
        
        # Get image configuration for this output type/platform
        config = self._get_image_config(output_type, platform)
        
        # Extract content information for prompt generation
        title = self._get_title(enriched_content)
        summary = self._get_summary(enriched_content)
        topics = self._get_topics(enriched_content)
        tags = self._get_tags(enriched_content)
        
        # Generate prompts based on output type
        prompts = self._generate_prompts_for_type(
            output_type=output_type,
            platform=platform,
            config=config,
            title=title,
            summary=summary,
            topics=topics,
            tags=tags,
        )
        
        return ImagePromptsResult(
            prompts=prompts,
            output_type=output_type,
            platform=platform,
            content_title=title,
        )
    
    def _get_image_config(self, output_type: str, platform: Optional[str]) -> dict:
        """Get image configuration for output type and platform.
        
        Args:
            output_type: The format output type
            platform: Optional target platform
            
        Returns:
            Configuration dictionary with image counts
        """
        # Check platform-specific config first
        if platform and platform in PLATFORM_IMAGE_COUNTS:
            return PLATFORM_IMAGE_COUNTS[platform]
        
        # Fall back to output type defaults
        default_key = f"{output_type}_default"
        if default_key in PLATFORM_IMAGE_COUNTS:
            return PLATFORM_IMAGE_COUNTS[default_key]
        
        # Generic fallback
        return {"header": 1, "supporting": 2, "total_max": 3}
    
    def _get_title(self, enriched_content: dict) -> str:
        """Extract title from enriched content."""
        metadata = enriched_content.get("metadata", {})
        return metadata.get("title", "Untitled Content")
    
    def _get_summary(self, enriched_content: dict) -> str:
        """Extract summary from enriched content."""
        summary = enriched_content.get("summary", {})
        if isinstance(summary, dict):
            return summary.get("medium", summary.get("short", summary.get("long", "")))
        return str(summary) if summary else ""
    
    def _get_topics(self, enriched_content: dict) -> list[str]:
        """Extract topics from enriched content."""
        topics = enriched_content.get("topics", [])
        return topics if isinstance(topics, list) else []
    
    def _get_tags(self, enriched_content: dict) -> list[str]:
        """Extract tags from enriched content."""
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            return tags.get("primary", []) + tags.get("secondary", [])
        return tags if isinstance(tags, list) else []
    
    def _generate_prompts_for_type(
        self,
        output_type: str,
        platform: Optional[str],
        config: dict,
        title: str,
        summary: str,
        topics: list[str],
        tags: list[str],
    ) -> list[ImagePrompt]:
        """Generate prompts based on output type.
        
        Args:
            output_type: Target output format
            platform: Target platform
            config: Image configuration
            title: Content title
            summary: Content summary
            topics: Content topics
            tags: Content tags
            
        Returns:
            List of ImagePrompt objects
        """
        prompts: list[ImagePrompt] = []
        topic_str = ", ".join(topics[:3]) if topics else "general topic"
        tag_str = ", ".join(tags[:5]) if tags else ""
        
        if output_type in {"blog", "tutorial"}:
            prompts.extend(self._generate_blog_prompts(
                config, title, summary, topic_str, tag_str
            ))
        elif output_type == "linkedin":
            prompts.extend(self._generate_linkedin_prompts(
                title, summary, topic_str
            ))
        elif output_type == "youtube":
            prompts.extend(self._generate_youtube_prompts(
                title, summary, topic_str
            ))
        elif output_type == "slides":
            prompts.extend(self._generate_slides_prompts(
                config, title, topics
            ))
        elif output_type == "newsletter":
            prompts.extend(self._generate_newsletter_prompts(
                config, title, topic_str
            ))
        
        return prompts
    
    def _generate_blog_prompts(
        self,
        config: dict,
        title: str,
        summary: str,
        topic_str: str,
        tag_str: str,
    ) -> list[ImagePrompt]:
        """Generate prompts for blog/tutorial content."""
        prompts = []
        
        # Header image
        if config.get("header", 0) > 0:
            prompts.append(ImagePrompt(
                prompt=f"A modern, professional header image for a blog article about {topic_str}. "
                       f"Title: '{title}'. Clean design with subtle tech elements, "
                       f"gradient background in blue and purple tones, minimalist style, "
                       f"suitable for Medium or professional blog. No text in image.",
                image_type="header",
                suggested_dimensions=DEFAULT_DIMENSIONS["header"],
                position_hint="article header",
                style_notes="Professional, clean, tech-focused"
            ))
        
        # Supporting images (infographics, diagrams)
        supporting_count = config.get("supporting", 0)
        if supporting_count > 0:
            # First supporting: infographic
            prompts.append(ImagePrompt(
                prompt=f"An informative infographic visualizing key concepts about {topic_str}. "
                       f"Show relationships and flow between main ideas. "
                       f"Modern flat design, clean icons, clear visual hierarchy. "
                       f"Color scheme: professional blues and greens.",
                image_type="infographic",
                suggested_dimensions=DEFAULT_DIMENSIONS["infographic"],
                position_hint="after introduction section",
                style_notes="Educational, clear labels, modern flat design"
            ))
        
        if supporting_count > 1:
            # Second supporting: diagram
            prompts.append(ImagePrompt(
                prompt=f"A technical diagram showing the architecture or process flow for {topic_str}. "
                       f"Clean vector style with labeled components, arrows showing data flow. "
                       f"White background, professional appearance.",
                image_type="diagram",
                suggested_dimensions=DEFAULT_DIMENSIONS["diagram"],
                position_hint="technical explanation section",
                style_notes="Technical, clear, labeled components"
            ))
        
        if supporting_count > 2:
            # Additional illustrations
            for i in range(supporting_count - 2):
                prompts.append(ImagePrompt(
                    prompt=f"An illustration representing a key concept from {topic_str}. "
                           f"Modern, friendly style with soft colors. "
                           f"Abstract representation suitable for tech content.",
                    image_type="illustration",
                    suggested_dimensions=DEFAULT_DIMENSIONS["illustration"],
                    position_hint=f"section {i + 3}",
                    style_notes="Friendly, modern, abstract"
                ))
        
        return prompts
    
    def _generate_linkedin_prompts(
        self,
        title: str,
        summary: str,
        topic_str: str,
    ) -> list[ImagePrompt]:
        """Generate prompts for LinkedIn content."""
        return [
            ImagePrompt(
                prompt=f"A professional, corporate-style header image for LinkedIn post about {topic_str}. "
                       f"Clean, modern design with subtle business elements. "
                       f"Blue and white color scheme, minimalist aesthetic. "
                       f"Conveys expertise and professionalism. No text overlay.",
                image_type="header",
                suggested_dimensions=(1200, 627),  # LinkedIn recommended
                position_hint="post header",
                style_notes="Professional, corporate, LinkedIn-optimized"
            )
        ]
    
    def _generate_youtube_prompts(
        self,
        title: str,
        summary: str,
        topic_str: str,
    ) -> list[ImagePrompt]:
        """Generate prompts for YouTube content."""
        return [
            ImagePrompt(
                prompt=f"An eye-catching YouTube thumbnail for video about {topic_str}. "
                       f"Bold, vibrant colors with high contrast. "
                       f"Dynamic composition that grabs attention. "
                       f"Space on right side for text overlay. "
                       f"Professional but energetic style.",
                image_type="thumbnail",
                suggested_dimensions=DEFAULT_DIMENSIONS["thumbnail"],
                position_hint="video thumbnail",
                style_notes="Bold, high-contrast, attention-grabbing"
            ),
            ImagePrompt(
                prompt=f"A YouTube end screen background for video about {topic_str}. "
                       f"Clean design with space for subscribe button and video cards. "
                       f"Matching style to thumbnail, cohesive branding. "
                       f"Dark background with accent colors.",
                image_type="end_screen",
                suggested_dimensions=DEFAULT_DIMENSIONS["end_screen"],
                position_hint="video end screen",
                style_notes="Clean, space for overlays, cohesive branding"
            )
        ]
    
    def _generate_slides_prompts(
        self,
        config: dict,
        title: str,
        topics: list[str],
    ) -> list[ImagePrompt]:
        """Generate prompts for slide presentations."""
        prompts = []
        
        # Title slide
        prompts.append(ImagePrompt(
            prompt=f"A professional presentation title slide background for '{title}'. "
                   f"Modern gradient design, subtle geometric patterns. "
                   f"Dark theme with accent colors. Space for title text.",
            image_type="header",
            suggested_dimensions=(1920, 1080),
            position_hint="title slide",
            style_notes="Presentation, dark theme, modern"
        ))
        
        # Concept illustrations for each topic
        concept_count = min(config.get("concepts", 3), len(topics) if topics else 3)
        for i, topic in enumerate(topics[:concept_count]):
            prompts.append(ImagePrompt(
                prompt=f"An illustration representing '{topic}' for a presentation slide. "
                       f"Clean, minimal design with single focal point. "
                       f"Consistent style, professional appearance.",
                image_type="illustration",
                suggested_dimensions=(800, 600),
                position_hint=f"slide for topic: {topic}",
                style_notes="Minimal, focused, presentation-ready"
            ))
        
        return prompts
    
    def _generate_newsletter_prompts(
        self,
        config: dict,
        title: str,
        topic_str: str,
    ) -> list[ImagePrompt]:
        """Generate prompts for newsletter content."""
        prompts = []
        
        # Banner
        prompts.append(ImagePrompt(
            prompt=f"A newsletter header banner about {topic_str}. "
                   f"Wide format, clean design with brand-friendly colors. "
                   f"Professional but approachable style. "
                   f"Space for newsletter title overlay.",
            image_type="banner",
            suggested_dimensions=DEFAULT_DIMENSIONS["banner"],
            position_hint="newsletter header",
            style_notes="Wide format, brand-friendly, professional"
        ))
        
        # Section dividers
        divider_count = config.get("dividers", 2)
        for i in range(divider_count):
            prompts.append(ImagePrompt(
                prompt=f"A decorative section divider for newsletter about {topic_str}. "
                       f"Subtle, elegant design. Horizontal format. "
                       f"Matches newsletter branding, not distracting.",
                image_type="divider",
                suggested_dimensions=DEFAULT_DIMENSIONS["divider"],
                position_hint=f"section divider {i + 1}",
                style_notes="Subtle, decorative, horizontal"
            ))
        
        return prompts
    
    def to_json(self, result: ImagePromptsResult) -> str:
        """Convert result to JSON string.
        
        Args:
            result: ImagePromptsResult to convert
            
        Returns:
            JSON string representation
        """
        data = {
            "schema_version": result.schema_version,
            "content_title": result.content_title,
            "output_type": result.output_type,
            "platform": result.platform,
            "prompt_count": len(result.prompts),
            "prompts": [
                {
                    "prompt": p.prompt,
                    "image_type": p.image_type,
                    "suggested_dimensions": list(p.suggested_dimensions),
                    "position_hint": p.position_hint,
                    "style_notes": p.style_notes,
                }
                for p in result.prompts
            ]
        }
        return json.dumps(data, indent=2)
    
    def get_output_filename(self, base_output: str) -> str:
        """Generate the image prompts output filename.
        
        Args:
            base_output: The base output file path
            
        Returns:
            Image prompts filename (e.g., "blog-image-prompts.json")
        """
        path = Path(base_output)
        return str(path.parent / f"{path.stem}-image-prompts.json")
