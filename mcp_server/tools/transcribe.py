"""Transcribe tool — wraps the transcribe CLI command."""

import json
from typing import Optional


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

    Returns:
        Dict with success status and transcript path.
    """
    try:
        from pipeline.transcription.orchestrator import TranscriptionOrchestrator

        orchestrator = TranscriptionOrchestrator()
        result = orchestrator.transcribe(
            source=input_path,
            engine=engine,
            language=language,
            output_path=output_path,
        )

        return {
            "success": True,
            "output_path": result.get("output_path", output_path),
            "engine": engine,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
