"""Extract tool — wraps the extract CLI command."""

import json
import os
from typing import Optional

from pipeline.extractors.dispatch import classify_source
from pipeline.extractors.youtube.extractor import YouTubeExtractor
from pipeline.extractors.local.file_audio import extract_audio_from_file
from pipeline.extractors.schema.metadata import build_local_placeholder_metadata


async def extract(
    source: str,
    output_path: Optional[str] = None,
) -> dict:
    """Extract audio and metadata from a video URL or local file.

    Args:
        source: YouTube URL or local file path.
        output_path: Output audio file path (default: output/<filename>.mp3).

    Returns:
        Dict with success status, output paths, and metadata.
    """
    source_type = classify_source(source)
    os.makedirs("output", exist_ok=True)

    if not output_path:
        output_path = os.path.join("output", "extracted_audio.mp3")

    metadata_path = output_path.replace(".mp3", ".json")

    try:
        if source_type == "streaming":
            extractor = YouTubeExtractor()
            metadata = extractor.extract_metadata(source)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            extractor.extract_audio(source, output_path)
        elif source_type == "file_system":
            if not os.path.exists(source):
                return {"success": False, "error": f"Input file not found: {source}"}
            metadata = build_local_placeholder_metadata(source)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            extract_audio_from_file(source, output_path)
        else:
            return {"success": False, "error": f"Unsupported source type: {source_type}"}

        return {
            "success": True,
            "output_path": output_path,
            "metadata_path": metadata_path,
            "source_type": source_type,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
