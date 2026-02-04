## 🧭 System Architecture Overview

This document outlines the high-level architecture of the **Content Pipeline**, designed for scalable, multi-agent audio and metadata extraction. The current implementation supports YouTube, with a modular design that enables future integration of platforms like Vimeo and TikTok. It supports CLI orchestration, schema enforcement, and agent-based modularity.

> **Note on v0.7.5 Infrastructure Refactoring**: This document reflects the infrastructure refactoring completed in v0.7.5, which introduced enterprise-grade provider architecture for LLM and transcription services. This was an unplanned technical release to establish a solid foundation before continuing with planned feature development in v0.8.0. See [infrastructure-migration-guide.md](infrastructure-migration-guide.md) for migration details.

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
- `validate_requirements()` — Checks if engine dependencies and credentials are available

Enhanced in v0.6.5 with multiple engine support:
- `base.py` — Enhanced adapter protocol with cost estimation and capability reporting
- `local_whisper.py` — Local Whisper adapter for privacy-first transcription
- `openai_whisper.py` — OpenAI Whisper API adapter for cloud-based transcription
- `aws_transcribe.py` — AWS Transcribe adapter for enterprise transcription
- `whisper.py` — Backward compatibility adapter (deprecated, use `local_whisper.py`)
- `auto_selector.py` — Smart engine selection with intelligent fallback
- `factory.py` — Engine factory pattern for adapter instantiation

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

### 2.5 Configuration Management

Centralized configuration system introduced in v0.6.5 for managing transcription engines, API keys, and output preferences.

#### `pipeline/config/`
- `manager.py` — ConfigurationManager for loading and merging configuration sources
- `schema.py` — Pydantic models for configuration validation (TranscriptionConfig, EngineConfig)
- `environment.py` — Environment variable definitions and loading
- `yaml_parser.py` — YAML parsing with enhanced error reporting
- `pretty_printer.py` — Configuration template generation

Configuration sources (in precedence order):
1. CLI flags (highest priority)
2. Environment variables
3. Explicit config file (`--config file.yaml`)
4. Project config file (`./.content-pipeline/config.yaml`)
5. User config file (`~/.content-pipeline/config.yaml`)
6. Default values (lowest priority)

---

### 2.6 Output Management

Flexible output path management introduced in v0.6.5, replacing hardcoded output directories.

#### `pipeline/output/`
- `manager.py` — OutputManager for resolving and managing output paths
- Supports absolute paths, relative paths, and directory-based output
- Automatic directory creation and unique filename generation
- Integration with configuration system for default output directories

---

### 2.7 Infrastructure Layer

Enterprise-grade provider architecture introduced in v0.7.5 for unified LLM and transcription service management.

#### `pipeline/llm/` — LLM Provider Infrastructure

Unified infrastructure for all LLM providers with consistent interface and configuration management:

- **Base Protocol**: `BaseLLMProvider` defines the contract for all LLM providers
  - `generate()` — Generate text from prompts
  - `estimate_cost()` — Calculate cost before generation
  - `validate_requirements()` — Check provider availability

- **Providers**: Standardized naming pattern `{Deployment}{Service}Provider`
  - `LocalOllamaProvider` — Local Ollama models (zero cost)
  - `CloudOpenAIProvider` — OpenAI GPT models
  - `CloudAnthropicProvider` — Anthropic Claude models
  - `CloudAWSBedrockProvider` — AWS Bedrock (Claude and Titan)

- **Factory Pattern**: `LLMProviderFactory` with caching and auto-selection
  - Provider instantiation by name
  - Automatic provider selection based on availability
  - Provider caching to prevent redundant instantiation
  - Configuration validation before provider creation

- **Configuration**: `LLMConfig` with environment variable substitution
  - Configuration precedence: Explicit params > Environment vars > Project config > User config > Defaults
  - Environment variable substitution: `${VAR_NAME:-default}`
  - Provider-specific configs: `OllamaConfig`, `OpenAIConfig`, `BedrockConfig`, `AnthropicConfig`
  - YAML configuration support with validation

- **Error Hierarchy**: Comprehensive error classes
  - `LLMError` — Base exception for all LLM errors
  - `ConfigurationError` — Configuration validation failures
  - `ProviderError` — Provider-specific errors
  - `ProviderNotAvailableError` — Provider unavailable errors

#### `pipeline/transcription/` — Transcription Provider Infrastructure

Unified infrastructure for all transcription providers with consistent interface and configuration management:

- **Base Protocol**: `TranscriberProvider` defines the contract for all transcription providers
  - `transcribe()` — Convert audio to text
  - `validate_requirements()` — Check provider availability

- **Providers**: Standardized naming pattern `{Deployment}{Service}Provider`
  - `LocalWhisperProvider` — Local Whisper models (privacy-first)
  - `CloudOpenAIWhisperProvider` — OpenAI Whisper API
  - `CloudAWSTranscribeProvider` — AWS Transcribe (enterprise-grade)

- **Factory Pattern**: `TranscriptionProviderFactory` with caching and validation
  - Provider instantiation by name
  - Provider caching to prevent redundant instantiation
  - Configuration validation before provider creation
  - Requirement validation (API keys, credentials)

- **Configuration**: `TranscriptionConfig` with environment variable substitution
  - Configuration precedence: Explicit params > Environment vars > Project config > User config > Defaults
  - Environment variable substitution: `${VAR_NAME:-default}`
  - Provider-specific configs: `WhisperLocalConfig`, `WhisperAPIConfig`, `AWSTranscribeConfig`
  - YAML configuration support with validation

- **Error Hierarchy**: Comprehensive error classes
  - `TranscriptionError` — Base exception for all transcription errors
  - `ConfigurationError` — Configuration validation failures
  - `ProviderError` — Provider-specific errors
  - `ProviderNotAvailableError` — Provider unavailable errors
  - `AudioFileError` — Audio file processing errors
  - `TranscriptionTimeoutError` — Timeout errors

**Key Design Principles**:
- **Separation of Concerns**: LLM and transcription infrastructure are completely separate
- **Consistent Interface**: All providers implement the same protocol
- **Configuration Objects**: No individual parameters, only configuration objects
- **No Hardcoded Values**: All configuration externalized to YAML or environment variables
- **Factory Pattern**: Centralized provider instantiation with caching
- **Error Handling**: Comprehensive error hierarchy for debugging

#### Pricing Configuration

Configurable pricing system introduced in v0.7.5 for accurate cost estimation with custom pricing agreements.

**Transcription Provider Pricing**:
- Simple per-minute pricing model
- Configurable via `cost_per_minute_usd` field in provider configs
- Environment variable support: `WHISPER_API_COST_PER_MINUTE`, `AWS_TRANSCRIBE_COST_PER_MINUTE`
- Default values: OpenAI Whisper ($0.006/min), AWS Transcribe ($0.024/min)
- Use cases: Volume discounts, regional pricing, custom enterprise agreements

**LLM Provider Pricing**:
- Complex per-model, per-token pricing
- Configurable via `pricing_override` field (optional dictionary)
- Format: `{"model-name": {"input_per_1k": 0.01, "output_per_1k": 0.03}}`
- Falls back to built-in pricing database if not overridden
- Supports all models across OpenAI, Anthropic, and AWS Bedrock providers

**Configuration Methods**:
1. **YAML Configuration** (`.content-pipeline/config.yaml`):
   ```yaml
   whisper_api:
     cost_per_minute_usd: 0.005  # Custom rate
   
   llm:
     openai:
       pricing_override:
         gpt-4: {input_per_1k: 0.025, output_per_1k: 0.05}
   ```

2. **Environment Variables** (for transcription only):
   ```bash
   export WHISPER_API_COST_PER_MINUTE=0.005
   export AWS_TRANSCRIBE_COST_PER_MINUTE=0.020
   ```

3. **Configuration Precedence**: Environment > YAML > Default

**Benefits**:
- Enterprise customers can reflect negotiated pricing
- Easy updates when providers change rates
- No code changes required for pricing updates
- Maintains backward compatibility with default values

---

### 3. Enrichment System

AI-powered semantic enrichment introduced in v0.7.0 for transforming transcripts into structured, semantically rich content.

#### `pipeline/enrichment/`
Core enrichment infrastructure for LLM-powered analysis:

- `orchestrator.py` — Coordinates enrichment workflow across agents, prompts, and validation
- `cost_estimator.py` — Pre-flight cost calculation with token counting and pricing database
- `cache.py` — File-based caching system with TTL expiration and size limits
- `chunking.py` — Automatic transcript splitting for long-form content
- `batch.py` — Batch processing with progress tracking and error handling
- `validate.py` — Schema validation and automatic repair for LLM responses
- `retry.py` — Exponential backoff retry logic for transient failures
- `output.py` — Output file management with path resolution
- `errors.py` — Comprehensive error hierarchy for enrichment operations

---

#### `pipeline/enrichment/agents/`
LLM provider adapters implementing unified agent protocol:

- `base.py` — BaseLLMAgent protocol defining agent interface
- `openai_agent.py` — OpenAI GPT models (GPT-4, GPT-3.5-turbo)
- `claude_agent.py` — Anthropic Claude models (Claude 3 Opus/Sonnet/Haiku, Claude 2)
- `bedrock_agent.py` — AWS Bedrock (Claude and Titan models)
- `ollama_agent.py` — Local Ollama models (Llama 2, Mistral, etc.)
- `factory.py` — Agent factory with auto-selection and credential validation

All agents support:
- Cost estimation with provider-specific token counting
- Context window detection and validation
- Standardized request/response formats
- Retry logic with exponential backoff

---

#### `pipeline/enrichment/schemas/`
Pydantic models for enrichment output validation:

- `enrichment_v1.py` — EnrichmentV1 container with metadata
- `summary.py` — SummaryEnrichment (short/medium/long variants)
- `tag.py` — TagEnrichment (categories, keywords, entities)
- `chapter.py` — ChapterEnrichment (title, timestamps, description)
- `highlight.py` — HighlightEnrichment (quote, timestamp, importance level)

All schemas include:
- Field validation with Pydantic v2
- JSON schema generation
- Automatic repair logic for common LLM output issues

---

#### `pipeline/enrichment/prompts/`
YAML-based prompt engineering system:

- `loader.py` — PromptLoader for loading and caching templates
- `renderer.py` — PromptRenderer with Jinja2 templating
- `summarize.yaml` — Summary generation prompt
- `tag.yaml` — Tag extraction prompt
- `chapterize.yaml` — Chapter detection prompt
- `highlight.yaml` — Highlight identification prompt

Supports:
- Custom prompt directories
- Template variables (transcript_text, language, duration, word_count)
- Fallback from custom to default prompts

---

#### `pipeline/enrichment/presets/`
Quality and content profile configurations:

- `quality.py` — Quality presets (FAST, BALANCED, BEST)
- `content.py` — Content profiles (PODCAST, MEETING, LECTURE)

Quality presets select appropriate models per provider:
- FAST: Smaller, cheaper models (gpt-3.5-turbo, claude-haiku, llama2:7b)
- BALANCED: Mid-tier models (gpt-4-turbo, claude-sonnet, llama2:13b)
- BEST: Largest models (gpt-4, claude-opus, llama2:70b)

Content profiles adapt enrichment to domain:
- PODCAST: Medium summaries, speaker extraction, chapter detection
- MEETING: Short summaries, action items, decision highlights
- LECTURE: Long summaries, key concepts, chapter detection

---

### 4. CLI Orchestration

Modular CLI architecture refactored in v0.6.0 into the `cli/` package.

The CLI is organized into subcommands using Click groups with shared components:

- `extract` — triggers the extraction pipeline
- `transcribe` — triggers the transcription pipeline
- `enrich` — triggers the enrichment pipeline (NEW in v0.7.0)

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

Enhanced in v0.6.5 with engine selection and configuration:
- `--engine` — **REQUIRED** transcription engine selection (local-whisper, openai-whisper, aws-transcribe, auto)
- `--model` — model size/version for selected engine (e.g., `base`, `large`, `whisper-1`)
- `--api-key` — API key for cloud services (or use environment variables)
- `--config` — path to YAML configuration file
- `--output-dir` — output directory (overrides configuration)
- `--log-level` — logging verbosity (debug, info, warning, error)

Output includes:
- Transcript `.json` conforming to `TranscriptV1` schema

---

#### 🎨 Enrich Flags

Used with the `enrich` subcommand (NEW in v0.7.0):

- `--input` — path to transcript file or glob pattern for batch processing
- `--output` — path for saving enriched output (auto-generated if not specified)
- `--output-dir` — directory for batch processing outputs
- `--provider` — LLM provider selection (openai, claude, bedrock, ollama, auto)
- `--model` — specific model to use (overrides quality preset)
- `--quality` — quality preset (fast, balanced, best)
- `--preset` — content profile (podcast, meeting, lecture, custom)
- `--summarize` — generate summaries
- `--tag` — extract tags
- `--chapterize` — detect chapters
- `--highlight` — identify highlights
- `--all` — enable all enrichment types
- `--max-cost` — maximum cost limit in USD
- `--dry-run` — preview costs without making API calls
- `--no-cache` — bypass cache and generate fresh results
- `--custom-prompts` — directory with custom YAML prompt templates
- `--config` — path to configuration file
- `--log-level` — logging verbosity

Output includes:
- Enriched transcript `.json` conforming to `EnrichmentV1` schema
- Metadata including provider, model, cost, tokens, and cache status

---

Handles logging, error propagation, and output normalization across all flows.

---

### 5. Schema Enforcement

#### `pipeline/extractors/schema/metadata.py`

- Defines the metadata schema used by extractors (YouTube, local)
- Enforced via unit tests and integration validation
- Ensures consistent downstream consumption by agents or GUI

#### `pipeline/transcribers/schemas/transcript_v1.py`

- Defines the transcript schema used by transcriber adapters
- Enforced via integration tests and schema validation
- Enables structured enrichment, publishing, and archival

#### `pipeline/enrichment/schemas/`

- Defines enrichment output schemas (EnrichmentV1, Summary, Tag, Chapter, Highlight)
- Enforced via Pydantic validation with automatic repair
- Ensures consistent downstream formatting and publishing

---

### 6. Configuration & Logging

#### `pipeline/config/logging_config.py`

- Centralized logging configuration for CLI and pipeline modules  
- Supports structured logs, verbosity control, and test isolation

---

### 7. Utilities

#### `pipeline/utils/retry.py`

- Generic retry logic for transient failures (e.g., network, API)
- Used across extractors and CLI

---

## 8. Multi-Agent Protocol (Planned)

The pipeline will integrate with an MCP server to support agent-based orchestration:

- Agents will invoke extractors via CLI or direct module calls
- Metadata and audio outputs will be tagged and routed via shared schema
- Future support for GUI enrichment and real-time observability

---

## 9. Observability & Testing

- Integration tests validate CLI behavior and extractor output
- Logging is unified across all components
- Future plans include tracing and metrics for agent workflows

---

## 10. Test Coverage

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
  - `v0.6.5`: Enhanced transcription with multiple engines, configuration management, and flexible output paths
  - `v0.7.0`: LLM-powered enrichment with multi-provider support, cost control, and intelligent caching

---

## 📁 Folder Summary

For full folder and file layout, see [project_structure.md](project_structure.md)

---

## 🧭 Future Directions
- 📝 Format enriched outputs for publishing: blog drafts, tweet threads, chapters, and SEO tags across major social media platforms
- 📦 Archive and index all enriched content into a searchable store  
- 🧠 Integrate MCP server for agent orchestration, routing, retries, and tagging  
- 🖥️ Build a GUI for reviewing and editing enriched metadata before publishing  
- 📊 Add real-time observability: structured logging, tracing, and metrics across pipeline stages  


