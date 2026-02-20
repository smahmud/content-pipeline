"""
Content Pipeline MCP Server

Exposes pipeline commands as MCP tools for AI agent orchestration.
Supports stdio transport (for Kiro/Claude Desktop) and SSE transport.

Usage:
    python -m mcp_server.server                    # stdio mode (default)
    python -m mcp_server.server --transport sse     # SSE mode
"""

import argparse
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPServerConfig

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    "content-pipeline",
)


# --- Tool Definitions ---

@mcp.tool()
async def extract(
    source: str,
    output_path: Optional[str] = None,
) -> dict:
    """Extract audio and metadata from a video URL or local file.

    Args:
        source: YouTube URL or local file path.
        output_path: Output audio file path (default: output/extracted_audio.mp3).
    """
    from mcp_server.tools.extract import extract as _extract
    return await _extract(source=source, output_path=output_path)


@mcp.tool()
async def transcribe(
    input_path: str,
    engine: str = "local-whisper",
    language: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Transcribe audio to structured transcript.

    Args:
        input_path: Path to audio file.
        engine: Transcription engine (local-whisper, openai-whisper, aws-transcribe, auto).
        language: Language code (e.g., 'en').
        output_path: Output transcript file path.
    """
    from mcp_server.tools.transcribe import transcribe as _transcribe
    return await _transcribe(
        input_path=input_path, engine=engine,
        language=language, output_path=output_path,
    )


@mcp.tool()
async def enrich(
    input_path: str,
    provider: str = "auto",
    summarize: bool = True,
    tag: bool = True,
    chapterize: bool = True,
    highlight: bool = True,
    output_path: Optional[str] = None,
) -> dict:
    """Generate semantic enrichment from transcript using LLM.

    Args:
        input_path: Path to transcript JSON file.
        provider: LLM provider (auto, cloud-aws-bedrock, cloud-openai, cloud-anthropic, local-ollama).
        summarize: Generate summary enrichment.
        tag: Generate tag enrichment.
        chapterize: Generate chapter enrichment.
        highlight: Generate highlight enrichment.
        output_path: Output enriched JSON file path.
    """
    from mcp_server.tools.enrich import enrich as _enrich
    return await _enrich(
        input_path=input_path, provider=provider,
        summarize=summarize, tag=tag,
        chapterize=chapterize, highlight=highlight,
        output_path=output_path,
    )


@mcp.tool()
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
        output_type: Output format type (blog, tweet, linkedin, youtube, seo, newsletter, etc.).
        platform: Target platform for validation (twitter, linkedin, medium, etc.).
        tone: Tone override (professional, casual, technical, friendly).
        length: Length override (short, medium, long).
        style_profile: Path to style profile Markdown file.
        llm_enhance: Enable LLM enhancement (default: True).
        output_path: Output file path.
    """
    from mcp_server.tools.format import format_content as _format
    return await _format(
        input_path=input_path, output_type=output_type,
        platform=platform, tone=tone, length=length,
        style_profile=style_profile, llm_enhance=llm_enhance,
        output_path=output_path,
    )


@mcp.tool()
async def validate(
    input_path: str,
    schema: str = "auto",
    platform: Optional[str] = None,
    strict: bool = False,
) -> dict:
    """Validate pipeline artifacts against schemas and platform limits.

    Args:
        input_path: Path to JSON file to validate.
        schema: Schema type (auto, transcript, enrichment, format).
        platform: Optional platform to validate against (twitter, linkedin, etc.).
        strict: Fail on warnings in addition to errors.
    """
    from mcp_server.tools.validate import validate as _validate
    return await _validate(
        input_path=input_path, schema=schema,
        platform=platform, strict=strict,
    )


@mcp.tool()
async def run_pipeline(
    input_url: str,
    output_types: Optional[List[str]] = None,
    platform: Optional[str] = None,
) -> dict:
    """Run full pipeline: extract → transcribe → enrich → format.

    Args:
        input_url: YouTube URL or local file path.
        output_types: List of output format types (default: ["blog"]).
        platform: Target platform for format validation.
    """
    from mcp_server.tools.pipeline import run_pipeline as _pipeline
    return await _pipeline(
        input_url=input_url, output_types=output_types,
        platform=platform,
    )


# --- Server Entry Point ---

def main():
    """Start the MCP server."""
    parser = argparse.ArgumentParser(description="Content Pipeline MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=str, help="Path to server config YAML")
    args = parser.parse_args()

    config = MCPServerConfig.load(args.config)

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.WARNING))

    transport = args.transport or config.transport

    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
