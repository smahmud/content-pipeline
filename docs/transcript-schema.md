# Transcript Schema Specification

This document defines the normalized transcript structure used across all transcribers in the pipeline. It ensures consistency, readability, and downstream compatibility for enrichment, summarization, and indexing.

---

## 🧱 TranscriptV1 Fields
```test
| Field       | Type           | Description |
|-------------|----------------|-------------|
| `text`      | `str`          | Fully normalized transcript text |
| `language`  | `str`          | ISO 639-1 language code (e.g. `"en"`, `"fr"`) |
| `segments`  | `list[dict]`   | Optional list of timestamped segments (if available) |
```

---

## ✨ Normalization Rules

The `TranscriptV1` model applies the following transformations to raw transcriber output:

- Adds punctuation and capitalization
- Removes filler words and hesitations (e.g. "um", "uh")
- Normalizes whitespace and line breaks
- Ensures UTF-8 encoding and safe serialization
- Trims leading/trailing silence or noise artifacts

---

## 🧪 Example Output

```json
{
  "text": "Welcome to the channel. Today we're exploring the future of AI in education.",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Welcome to the channel."
    },
    {
      "start": 3.2,
      "end": 7.5,
      "text": "Today we're exploring the future of AI in education."
    }
  ]
}
```
---

## 🔌 Integration Points

- Used in the `transcribe` CLI command after Whisper transcription
- Output is persisted as `.txt` (flattened `text`) and optionally `.json` (full structure)
- Compatible with downstream enrichment providers and summarizers
- Normalization is handled by `normalize.py` using the `TranscriptV1` model

---

## ✅ Validation Rules

- All `TranscriptV1` objects must include a non-empty `text` field
- If `segments` are present, each must include `start`, `end`, and `text`
- `language` must conform to ISO 639-1 two-letter codes
- Segment timestamps must be non-negative and strictly increasing
