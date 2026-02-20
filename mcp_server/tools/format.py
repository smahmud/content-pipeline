"""Format tool — wraps the format CLI command."""

import json
from pathlib import Path
from typing import Optional


async def format_content(
    input_path: str,
    output_type: str,
    platform: Optional[str] = None,
    tone: Optional[str] = None,
    length: Optional[str] = None,
    style_profile: Optional[str] = None,
    llm_enhance: bool = True,
    output_path: Optional[str] = None,
) -> dict:
    """Format enriched content into publishing-ready output.

    Args:
        input_path: Path to enriched JSON file.
        output_type: Output format type (blog, tweet, linkedin, youtube, etc.).
        platform: Target platform for validation.
        tone: Tone override (professional, casual, technical, friendly).
        length: Length override (short, medium, long).
        style_profile: Path to style profile Markdown file.
        llm_enhance: Enable LLM enhancement.
        output_path: Output file path.

    Returns:
        Dict with success status and output details.
    """
    try:
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import FormatRequest
        from pipeline.formatters.bundles.loader import BundleLoader
        from pipeline.formatters.writer import OutputWriter

        # Load enriched content
        with open(input_path, "r", encoding="utf-8") as f:
            enriched_content = json.load(f)

        # Load style profile if specified
        style_profile_data = None
        if style_profile:
            from pipeline.formatters.style_profile import StyleProfileLoader
            loader = StyleProfileLoader()
            profile = loader.load(style_profile)
            style_profile_data = profile.__dict__ if profile else None

        # Initialize LLM enhancer if needed
        llm_enhancer_instance = None
        if llm_enhance:
            from pipeline.formatters.llm.enhancer import LLMEnhancer
            from pipeline.llm import LLMProviderFactory, LLMConfig

            llm_config = LLMConfig.load_from_yaml(".content-pipeline/config.yaml")
            provider_factory = LLMProviderFactory(llm_config)
            llm_enhancer_instance = LLMEnhancer(provider_factory=provider_factory)

        bundle_loader = BundleLoader()
        composer = FormatComposer(
            bundle_loader=bundle_loader,
            llm_enhancer=llm_enhancer_instance,
        )

        request = FormatRequest(
            enriched_content=enriched_content,
            output_type=output_type,
            platform=platform,
            style_profile=style_profile_data,
            tone=tone,
            length=length,
            llm_enhance=llm_enhance,
        )

        result = composer.format_single(request)

        if not result.success:
            return {"success": False, "error": result.error}

        # Write output
        writer = OutputWriter(force_overwrite=True)
        write_result = writer.write(
            format_result=result,
            output_path=output_path,
            input_path=input_path,
            embed_metadata=True,
            force=True,
        )

        return {
            "success": True,
            "output_path": write_result.output_path if write_result.success else output_path,
            "output_type": output_type,
            "warnings": result.warnings,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
