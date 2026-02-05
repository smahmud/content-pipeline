# Metadata Schema Specification

This document defines the canonical metadata structure used across all extractors in the pipeline. It ensures consistency, auditability, and extensibility for downstream providers and enrichment workflows.

---

## 🧱 Base Metadata Fields

These fields are required across all sources (`streaming`, `storage`, `file_system`):

| Field             | Type                                  | Description |
|------------------|---------------------------------------|-------------|
| `title`          | `str`                                 | Human-readable title of the content |
| `duration`       | `Optional[int]`                       | Duration in seconds (if available) |
| `author`         | `Optional[str]`                       | Creator or uploader name |
| `source_type`    | `Literal["streaming", "storage", "file_system"]` | Structural classification of the source |
| `source_path`    | `Optional[str]`                       | Absolute path to local file (if applicable) |
| `source_url`     | `Optional[str]`                       | Original URL of the content (if applicable) |
| `metadata_status`| `Literal["complete", "incomplete"]`   | Indicates whether metadata is fully enriched |
| `service_metadata` | `dict`                              | Optional service-specific fields (see below) |

---

## 🧩 `service_metadata` Extensions

This field contains optional, service-specific metadata. It is empty for local files and populated when available from streaming services.

### Example: YouTube

```json
"service_metadata": {
  "view_count": 12345,
  "channel_id": "UCabc123",
  "like_count": 678
}
```

---

## 🧪 Example Output

### YouTube Video Metadata
```json
{
  "title": "Introduction to Machine Learning",
  "duration": 1800,
  "author": "Tech Channel",
  "source_type": "streaming",
  "source_path": null,
  "source_url": "https://youtube.com/watch?v=example",
  "metadata_status": "complete",
  "service_metadata": {
    "view_count": 50000,
    "channel_id": "UCexample123",
    "like_count": 1200
  }
}
```

### Local File Metadata
```json
{
  "title": "recording_2024_01_15",
  "duration": 3600,
  "author": null,
  "source_type": "file_system",
  "source_path": "/path/to/recording.mp4",
  "source_url": null,
  "metadata_status": "incomplete",
  "service_metadata": {}
}
```

---

## 🔌 Integration Points

- Used by all extractors (`YouTubeExtractor`, `LocalFileExtractor`) to generate consistent metadata
- Persisted as `.json` files alongside extracted audio
- Consumed by CLI commands for display and logging
- Referenced by enrichment providers for context

---

## ✅ Validation Rules

- `title` must be a non-empty string
- `source_type` must be one of: `streaming`, `storage`, `file_system`
- Either `source_path` or `source_url` must be provided (not both)
- `metadata_status` indicates completeness: `complete` (all fields populated), `incomplete` (minimal fields only)
- `service_metadata` is optional and service-specific

---

## See Also

- **[Transcript Schema](transcript-schema.md)** - Transcript structure specification
- **[Architecture](architecture.md)** - System design and extractor overview
- **[CLI Commands](cli-commands.md)** - Command-line interface documentation
