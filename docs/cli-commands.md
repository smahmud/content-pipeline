# CLI Commands Reference

This document provides usage reference for the Content Pipeline CLI commands.

## 📊 Command Status

| Command | Status | Version | Description |
|---------|--------|---------|-------------|
| `extract` | ✅ Implemented | v0.1.0-v0.4.0 | Extract audio from video sources |
| `transcribe` | ✅ Implemented | v0.5.0-v0.6.5 | Convert audio to text transcripts |
| `enrich` | ✅ Implemented | v0.7.0 | Generate semantic metadata with LLM |

---

## ✅ **Implemented Commands**

**Note:** v0.6.0 focused on CLI refactoring - transforming monolithic CLI into modular architecture. No new functionality was added to existing commands, only structural improvements for maintainability and extensibility.

### `extract`
**Status:** ✅ Implemented (v0.1.0 - v0.4.0), Refactored in v0.6.0  
**Purpose:** Extracts audio and metadata from YouTube URLs, local video files, and streaming platforms.

**Usage:**
```bash
content-pipeline extract --source VIDEO_SOURCE --output AUDIO_FILE [OPTIONS]
```

**Key Options:**
- `--source, -s`: Input video source (YouTube URL, local file, etc.)
- `--output, -o`: Output audio file path
- `--format`: Audio format (mp3, wav, etc.)
- `--quality`: Audio quality settings

**Example:**
```bash
content-pipeline extract --source "https://youtube.com/watch?v=abc123" --output ./audio.mp3
content-pipeline extract --source ./video.mp4 --output ./audio.wav --format wav
```

### `transcribe`
**Status:** ✅ Implemented in v0.5.0, Refactored in v0.6.0, Enhanced in v0.6.5  
**Purpose:** Converts audio files to text transcripts using multiple transcription providers with configuration management.

**Usage:**
```bash
content-pipeline transcribe --source AUDIO_FILE --engine ENGINE_TYPE [OPTIONS]
```

**Key Options:**
- `--source, -s`: Input audio file path
- `--output, -o`: Output transcript file path
- `--engine`: **REQUIRED** - Transcription provider (local-whisper, openai-whisper, aws-transcribe, auto)
- `--language, -l`: Source language (auto-detect if not specified)
- `--model`: Model size/version for selected provider
- `--api-key`: API key for cloud services
- `--config`: Path to YAML configuration file
- `--output-dir`: Output directory (overrides configuration)
- `--log-level`: Logging verbosity (debug, info, warning, error)

**Examples:**

**v0.5.0-v0.6.0 (Legacy):**
```bash
content-pipeline transcribe --source ./audio.mp3 --output ./transcript.json --language en
```

**v0.6.5 Examples:**
```bash
# Local Whisper (privacy-first)
content-pipeline transcribe --source audio.mp3 --engine local-whisper --model base

# OpenAI API (quality-first)
content-pipeline transcribe --source audio.mp3 --engine openai-whisper --api-key $OPENAI_API_KEY

# AWS Transcribe (enterprise)
content-pipeline transcribe --source audio.mp3 --engine aws-transcribe

# Auto provider selection
content-pipeline transcribe --source audio.mp3 --engine auto --config ~/.content-pipeline/config.yaml

# Custom output directory
content-pipeline transcribe --source audio.mp3 --engine local-whisper --output-dir ./my-transcripts
```

**Enhanced in v0.6.5:**
- ✅ Multiple provider support (local-whisper, openai-whisper, aws-transcribe, auto)
- ✅ Explicit provider selection via `--engine` flag (REQUIRED)
- ✅ YAML configuration file support
- ✅ Environment variable integration for API keys
- ✅ User-controlled output paths
- ✅ Breaking change: `--engine` flag is now required

---

### `enrich`
**Status:** ✅ Implemented in v0.7.0  
**Purpose:** Generates semantic metadata including summaries, tags, chapters, and key highlights using LLM processing with multi-provider support.

**⚠️ Deprecation Notice**: The `--all` flag is deprecated and will be removed in v0.8.0. It has provider-specific reliability issues (only works with Anthropic Claude, fails with AWS Bedrock and OpenAI). Use separate commands for each enrichment type instead. See [deprecation notice](notes/all-flag-deprecation.md) for migration guide.

**Usage:**
```bash
content-pipeline enrich --input TRANSCRIPT_FILE [OPTIONS]
```

**Key Options:**
- `--input, -i`: Input transcript file or glob pattern for batch processing
- `--output, -o`: Output file path (auto-generated if not specified)
- `--output-dir`: Output directory for batch processing
- `--provider`: LLM provider (openai, claude, bedrock, ollama, auto)
- `--model`: Specific model to use (overrides quality preset)
- `--quality`: Quality preset (fast, balanced, best)
- `--preset`: Content profile (podcast, meeting, lecture, custom)
- `--summarize`: Generate summaries
- `--tag`: Extract tags
- `--chapterize`: Detect chapters
- `--highlight`: Identify highlights
- `--all`: **[DEPRECATED]** Enable all enrichment types (will be removed in v0.8.0 - use separate flags instead)
- `--max-cost`: Maximum cost limit in USD
- `--dry-run`: Preview costs without making API calls
- `--no-cache`: Bypass cache and generate fresh results
- `--custom-prompts`: Directory with custom YAML prompt templates
- `--config`: Path to configuration file
- `--log-level`: Logging verbosity

**Examples:**

**Recommended: Separate commands for each enrichment type:**
```bash
# Generate summary
content-pipeline enrich --input transcript.json --summarize

# Generate tags
content-pipeline enrich --input transcript.json --tag

# Generate chapters
content-pipeline enrich --input transcript.json --chapterize

# Generate highlights
content-pipeline enrich --input transcript.json --highlight
```

**Combine multiple enrichment types:**
```bash
content-pipeline enrich --input transcript.json --summarize --tag
```

**Provider selection:**
```bash
# OpenAI (requires API key)
content-pipeline enrich --input transcript.json --provider openai --summarize

# Local Ollama (free, privacy-first)
content-pipeline enrich --input transcript.json --provider ollama --summarize

# Auto-select best available
content-pipeline enrich --input transcript.json --provider auto --summarize
```

**Quality presets:**
```bash
# Fast and cheap
content-pipeline enrich --input transcript.json --quality fast --summarize

# Balanced (default)
content-pipeline enrich --input transcript.json --quality balanced --summarize

# Best quality
content-pipeline enrich --input transcript.json --quality best --summarize
```

**Content profiles:**
```bash
# Podcast profile (medium summaries, chapters, highlights)
content-pipeline enrich --input transcript.json --preset podcast

# Meeting profile (short summaries, action items)
content-pipeline enrich --input transcript.json --preset meeting

# Lecture profile (long summaries, key concepts)
content-pipeline enrich --input transcript.json --preset lecture
```

**Cost control:**
```bash
# Set maximum cost limit
content-pipeline enrich --input transcript.json --summarize --max-cost 0.50

# Dry run to preview costs
content-pipeline enrich --input transcript.json --summarize --tag --dry-run
```

**Batch processing:**
```bash
# Process all transcripts in directory
content-pipeline enrich --input "transcripts/*.json" --summarize --output-dir enriched/

# Process with cost limit
content-pipeline enrich --input "**/*.json" --summarize --tag --max-cost 5.00 --output-dir enriched/
```

**Custom prompts:**
```bash
# Use custom prompt templates
content-pipeline enrich --input transcript.json --summarize --custom-prompts ./my-prompts/
```

**Key Features in v0.7.0:**
- ✅ Multi-provider LLM support (OpenAI, Claude, Bedrock, Ollama)
- ✅ Four enrichment types (summary, tags, chapters, highlights)
- ✅ Cost estimation and control with `--max-cost` and `--dry-run`
- ✅ Quality presets (fast, balanced, best)
- ✅ Content profiles (podcast, meeting, lecture)
- ✅ Intelligent caching with TTL and size limits
- ✅ Batch processing with progress tracking
- ✅ Long transcript handling with automatic chunking
- ✅ Custom YAML prompt templates
- ✅ Retry logic with exponential backoff
- ⚠️ **Note**: The `--all` flag is deprecated due to provider-specific reliability issues. Use separate flags (e.g., `--summarize --tag`) or run separate commands for each enrichment type. See [deprecation notice](notes/all-flag-deprecation.md) for details.

---

## 🔄 **Command Workflow**

The commands work together in a content pipeline workflow:

```mermaid
graph LR
    A[Video Source] --> B[extract]
    B --> C[Audio File]
    C --> D[transcribe]
    D --> E[Transcript]
    E --> F[enrich]
    F --> G[Enriched Metadata]
```

**Current Workflow (v0.7.0):**
1. `extract` - Get audio from video source
2. `transcribe` - Convert audio to text transcript
3. `enrich` - Add semantic metadata (summaries, tags, chapters, highlights)

---

## 📖 **Related Documentation**

- [Installation Guide](installation-guide.md) - Dependency setup and validation
- [Configuration Guide](configuration-guide.md) - Configuration management and API keys
- [Architecture Documentation](architecture.md) - System design and patterns
