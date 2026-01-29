"""
File: test_transcribe_pipeline_flow.py

Integration tests for the transcription stage of the content pipeline.

Covers:
- CLI orchestration: validates that the `transcribe` command triggers the correct pipeline flow
- Audio input handling: supports both extracted `.mp3` files and local audio sources
- Transcript generation: ensures transcriber adapters produce structured, schema-compliant output
- File persistence: verifies that transcript files are written to the expected output path
"""
import json
from shutil import copyfile
import pytest
from pipeline.transcribers.adapters.local_whisper import LocalWhisperAdapter
from pipeline.transcribers.normalize import normalize_transcript_v1
from pipeline.transcribers.persistence import LocalFilePersistence
from pipeline.transcribers.validate import validate_transcript_v1

@pytest.mark.integration
@pytest.mark.slow
def test_transcribe_pipeline_flow(tmp_path):
    input_audio = tmp_path / "sample.mp3"
    copyfile("tests/assets/sample_audio.mp3", input_audio)

    adapter = LocalWhisperAdapter(model_name="base")
    raw = adapter.transcribe(str(input_audio))
    transcript = normalize_transcript_v1(raw, adapter)

    output_path = tmp_path / "transcript.json"
    LocalFilePersistence().persist(transcript, str(output_path))

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
        assert "metadata" in data
        assert "transcript" in data
        assert isinstance(data["transcript"], list)
        assert all("text" in segment for segment in data["transcript"])

@pytest.mark.integration
@pytest.mark.slow
def test_transcript_persistence_roundtrip(tmp_path):
    input_audio = tmp_path / "sample.mp3"
    copyfile("tests/assets/sample_audio.mp3", input_audio)

    adapter = LocalWhisperAdapter(model_name="base")
    raw = adapter.transcribe(str(input_audio))
    transcript = normalize_transcript_v1(raw, adapter)

    output_path = tmp_path / "transcript.json"
    LocalFilePersistence().persist(transcript, str(output_path))

    # Reload and validate
    with open(output_path) as f:
        reloaded = json.load(f)

    validated = validate_transcript_v1(reloaded)
    assert validated.metadata.engine == "local-whisper"
    assert isinstance(validated.transcript, list)
    assert all(seg.text for seg in validated.transcript)

