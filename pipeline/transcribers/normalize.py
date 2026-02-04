"""
File: normalize.py

Provides normalization utilities to convert raw transcript output into TranscriptV1 format.

DEPRECATED: This module is deprecated. Use pipeline.transcription providers directly.
"""
from pipeline.transcription import TranscriberProvider
from pipeline.transcribers.schemas.transcript_v1 import TranscriptV1, TranscriptSegment, build_transcript_metadata

def normalize_transcript_v1(raw: dict, adapter: TranscriberProvider) -> TranscriptV1:
    """
    Normalize a raw transcript dictionary into a TranscriptV1 object using adapter metadata.
    
    DEPRECATED: Use provider methods directly instead.
    """
    engine, version = adapter.get_engine_info()

    # Aggregate confidence scores
    confidences = [s.get("confidence") for s in raw.get("segments", []) if s.get("confidence") is not None]
    confidence_avg = round(sum(confidences) / len(confidences), 3) if confidences else None

    # Build metadata
    metadata = build_transcript_metadata(
        engine=engine,
        engine_version=version,
        language=raw.get("language"),
        confidence_avg=confidence_avg
    )

    # Build transcript segments
    segments = []
    for segment in raw.get("segments", []):
        segments.append(TranscriptSegment(
            text=segment["text"],
            timestamp=format_timestamp(segment["start"]),
            confidence=segment.get("confidence", None)
        ))

    return TranscriptV1(metadata=metadata, transcript=segments)

def format_timestamp(seconds: float) -> str:
    """
    Convert float seconds to HH:MM:SS.mmm format, clamping negatives to zero
    """
    seconds = max(0.0, seconds)
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


