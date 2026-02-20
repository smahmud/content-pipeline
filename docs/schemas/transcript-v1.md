# TranscriptV1 Schema Reference

**Status:** Frozen as of v1.0.0 — no breaking changes allowed.

## Structure

```json
{
    "metadata": {
        "engine": "string (required)",
        "engine_version": "string (required)",
        "schema_version": "string (required, e.g. 'transcript_v1')",
        "created_at": "datetime (required)",
        "language": "string (optional)",
        "confidence_avg": "float (optional, 0.0-1.0)"
    },
    "transcript": [
        {
            "text": "string (required)",
            "timestamp": "string (required, HH:MM:SS.mmm)",
            "speaker": "string (optional)",
            "confidence": "float (optional, 0.0-1.0)"
        }
    ]
}
```

## Fields

### metadata
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| engine | string | yes | Transcription engine used |
| engine_version | string | yes | Engine version |
| schema_version | string | yes | Always "transcript_v1" |
| created_at | datetime | yes | UTC timestamp |
| language | string | no | Language code (e.g., "en") |
| confidence_avg | float | no | Average confidence score |

### transcript (array of segments)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | yes | Transcribed text |
| timestamp | string | yes | Format: HH:MM:SS.mmm |
| speaker | string | no | Speaker identifier |
| confidence | float | no | Segment confidence (0.0-1.0) |
