"""
File: test_extract_pipeline_flow.py

Integration tests for the audio and metadata extraction stages of the content pipeline.

Covers:
- YouTube extraction: downloads `.mp3` audio and generates metadata from a public video URL
- Local file extraction: converts `.mp4` to `.mp3` and generates placeholder metadata
- Schema validation: ensures metadata conforms to expected structure and completeness
- File persistence: verifies that audio and metadata files are created and contain valid content
"""
import os
import json
import pytest
from pipeline.extractors.youtube.extractor import YouTubeExtractor
from pipeline.extractors.local.file_audio import extract_audio_from_file
from pipeline.extractors.schema.metadata import build_local_placeholder_metadata

TEST_OUTPUT_DIR = "tests/output"

@pytest.mark.integration
@pytest.mark.external
def test_extract_audio_from_youtube_integration():
    source = "https://www.youtube.com/watch?v=p4SoSxyJ7Nc"
    output_path = os.path.join(TEST_OUTPUT_DIR, "test_download.mp3")
    metadata_path = output_path.replace(".mp3", ".json")
    
    # Ensure output directory exists
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    try:        
        extractor = YouTubeExtractor()

        # Step 1: Download audio
        result_path = extractor.extract_audio(source, output_path)
        assert result_path.endswith(".mp3")
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Step 2: Extract and save metadata
        metadata = extractor.extract_metadata(source)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Validate metadata file
        assert os.path.exists(metadata_path)

        # Validate core schema
        assert metadata["source_type"] == "streaming"
        assert metadata["source_url"] == source
        assert metadata["source_path"] is None
        assert metadata["metadata_status"] == "complete"
        assert isinstance(metadata["title"], str)
        assert isinstance(metadata["duration"], int)
        assert isinstance(metadata["author"], str)

        # Validate service metadata
        assert "service_metadata" in metadata
        assert isinstance(metadata["service_metadata"], dict)
        assert "view_count" in metadata["service_metadata"] or "channel_id" in metadata["service_metadata"]

        # Validate full schema
        expected_keys = {
            "title", "duration", "author",
            "source_type", "source_path", "source_url",
            "metadata_status", "service_metadata"
        }
        assert expected_keys.issubset(metadata.keys())

    finally:
        pass
        for path in [output_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)


@pytest.mark.integration
def test_extract_audio_from_file_integration():
    video_path = "tests/assets/sample_video.mp4"
    output_path = os.path.join(TEST_OUTPUT_DIR, "test_audio.mp3")
    
    # Ensure output directory exists
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    try:
        result_path = extract_audio_from_file(video_path, output_path)
        assert result_path.endswith(".mp3")

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        metadata_path = output_path.replace(".mp3", ".json")
        metadata = build_local_placeholder_metadata(video_path)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Validate metadata file
        assert os.path.exists(metadata_path)

        # Validate core schema
        assert metadata["title"] == "sample_video.mp4"
        assert metadata["source_type"] == "file_system"
        assert metadata["source_path"] == os.path.abspath(video_path)
        assert metadata["source_url"] is None
        assert metadata["metadata_status"] == "incomplete"

        # Validate service metadata
        assert metadata["service_metadata"] == {}

        # Validate full schema
        expected_keys = {
            "title", "duration", "author",
            "source_type", "source_path", "source_url",
            "metadata_status", "service_metadata"
        }
        assert expected_keys.issubset(metadata.keys())

    finally:
        for path in [output_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)
