## 🧭 System Architecture Overview

This document outlines the high-level architecture of the **Content Pipeline**, designed for scalable, multi-agent audio and metadata extraction. The current implementation supports YouTube, with a modular design that enables future integration of platforms like Vimeo and TikTok. It supports CLI orchestration, schema enforcement, and agent-based modularity.

---

## 🧩 Core Components

### 1. Extractors  
Platform-specific modules that handle audio and metadata extraction.

#### `pipeline/extractors/local/`  
Local file ingestion now uses the unified metadata schema and classification logic:

- `file_audio.py` — Handles local audio file conversion and ingestion  
- `metadata.py` — Builds placeholder metadata using `build_local_placeholder_metadata()`  
- `classify_source()` — Determines `source_type` for routing and metadata construction

> Local extraction now conforms to the shared schema in `pipeline/schema/metadata.py`.

---

#### `pipeline/extractors/base.py`  
Defines the `BaseExtractor` interface for platform-specific audio and metadata extraction:
- `extract_audio(source, output_path)` — Extracts audio from a media source into a local file  
- `extract_metadata(source)` — Returns structured metadata for downstream enrichment  
- Subclasses implement platform-specific logic (e.g., YouTube, TikTok, Vimeo)

---

#### `pipeline/extractors/youtube/`  
Streaming service extractors implement a shared interface (`BaseExtractor`) and rely on centralized metadata logic:
- `extractor.py` — Unified entry point for YouTube audio and metadata extraction  
- Uses `pipeline/schema/metadata.py` for schema enforcement and normalization  

---

### 2. Transcribers  
Modular adapters that convert extracted audio into structured transcript data.

#### `pipeline/transcribers/adapters/`  
Adapter implementations for different transcription engines. Each adapter conforms to a shared interface (`TranscriberAdapter`) and exposes:
- `transcribe()` — Converts audio file to raw transcript dictionary  
- `get_engine_info()` — Returns engine name and version for metadata construction  

Current implementation:
- `whisper.py` — Uses OpenAI Whisper for transcription; supports multiple model variants

---

#### `pipeline/transcribers/normalize.py`  
Normalizes raw transcript output into a structured `TranscriptV1` object:
- Applies punctuation, casing, and whitespace normalization
- Optionally preserves timestamped segments
- Constructs `TranscriptV1` via `normalize_transcript()` and `build_transcript_metadata()`

---

#### `pipeline/transcribers/schemas/transcript_v1.py`
Defines the `TranscriptV1` schema used across the pipeline:
- `TranscriptSegment` — Individual text segment with timestamp, speaker, and confidence  
- `TranscriptMetadata` — Engine, version, language, and creation timestamp  
- `TranscriptV1` — Full transcript object with metadata and segments

---

#### `pipeline/transcribers/validate.py`  
Validates raw transcript dictionaries against the `TranscriptV1` schema:
- Raises `TranscriptValidationError` on malformed input  
- Enforces timestamp format and confidence bounds  
- Rejects extra fields via `extra="forbid"` model config

---

#### `pipeline/transcribers/persistence.py`  
Handles transcript serialization and file output:
- Persists any `TranscriptV1` or compatible object to disk  
- Returns absolute path to saved file

---

### 3. CLI Orchestration

Modular CLI architecture refactored in v0.6.0 into the `cli/` package.

The CLI is organized into subcommands using Click groups with shared components:

- `extract` — triggers the extraction pipeline
- `transcribe` — triggers the transcription pipeline

Each subcommand is implemented as a separate module with reusable decorators and centralized help text.

---

#### 🎧 Extract Flags

Used with the `extract` subcommand:

- `--source` — input media path (YouTube URL or local `.mp4`)
- `--output` — directory for saving extracted `.mp3` and metadata `.json`

Output includes:
- `.mp3` audio file
- Metadata `.json` conforming to the unified schema

---

#### 📝 Transcribe Flags

Used with the `transcribe` subcommand:

- `--source` — path to the input audio file (`.mp3`)
- `--output` — path for saving transcript output (`.json`)
- `--language` — specifies spoken language in the audio (e.g., `en`, `fr`, `de`)

Output includes:
- Transcript `.json` conforming to `TranscriptV1` schema

---

Handles logging, error propagation, and output normalization across all flows.

---

### 4. Schema Enforcement

#### `pipeline/extractors/schema/metadata.py`

- Defines the metadata schema used by extractors (YouTube, local)
- Enforced via unit tests and integration validation
- Ensures consistent downstream consumption by agents or GUI

#### `pipeline/transcribers/schemas/transcript_v1.py`

- Defines the transcript schema used by transcriber adapters
- Enforced via integration tests and schema validation
- Enables structured enrichment, publishing, and archival

---

### 5. Configuration & Logging

#### `pipeline/config/logging_config.py`

- Centralized logging configuration for CLI and pipeline modules  
- Supports structured logs, verbosity control, and test isolation

---

### 6. Utilities

#### `pipeline/utils/retry.py`

- Generic retry logic for transient failures (e.g., network, API)
- Used across extractors and CLI

---

## 7. Multi-Agent Protocol (Planned)

The pipeline will integrate with an MCP server to support agent-based orchestration:

- Agents will invoke extractors via CLI or direct module calls
- Metadata and audio outputs will be tagged and routed via shared schema
- Future support for GUI enrichment and real-time observability

---

## 8. Observability & Testing

- Integration tests validate CLI behavior and extractor output
- Logging is unified across all components
- Future plans include tracing and metrics for agent workflows

---

## 9. Test Coverage

- Unit tests validate extractor logic, schema compliance, and CLI flag behavior
- Integration tests simulate real input scenarios across platforms and verify output normalization
- Test scaffolds mirror actual CLI invocation and project structure
- Future plans include protocol-level agent tests and error scenario validation

---

## 📦 Versioning Discipline

- Semantic versioning is enforced:
  - `v0.2.x`: CLI integration and logging hardening
  - `v0.3.x`: Metadata extraction and schema enforcement
  - `v0.4.x`: Architecture overhaul and multi-agent readiness
  - `v0.5.x`: Transcriber functionality with Whisper integration
  - `v0.6.0`: CLI refactoring with modular architecture

---

## 📁 Folder Summary

For full folder and file layout, see [project_structure.md](project_structure.md)

---

## 🧭 Future Directions
- 🧠 Enhanced transcription with explicit engine selection and configuration management
- 🤖 Summarize transcripts with LLMs to generate structured highlights, tags, and semantic metadata  
- 📝 Format enriched outputs for publishing: blog drafts, tweet threads, chapters, and SEO tags across major social media platforms
- 📦 Archive and index all enriched content into a searchable store  
- 🧠 Integrate MCP server for agent orchestration, routing, retries, and tagging  
- 🖥️ Build a GUI for reviewing and editing enriched metadata before publishing  
- 📊 Add real-time observability: structured logging, tracing, and metrics across pipeline stages  


