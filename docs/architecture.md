## 🧭 System Architecture Overview

This document outlines the high-level architecture of the **Content Pipeline**, designed for scalable, multi-agent audio and metadata extraction. The current implementation supports YouTube, with a modular design that enables future integration of platforms like Vimeo and TikTok. It supports CLI orchestration, schema enforcement, and agent-based modularity.

> **Note on v0.7.5 Infrastructure Refactoring**: This document reflects the infrastructure refactoring completed in v0.7.5, which introduced enterprise-grade provider architecture for LLM and transcription services. This was an unplanned technical release to establish a solid foundation before continuing with planned feature development in v0.8.0. See [infrastructure-migration-guide.md](infrastructure-migration-guide.md) for migration details.

---

## 🧩 Core Components

The Content Pipeline has three core components, each mapped directly to a CLI command:

1. **Extractors** (`extract` command) - Platform-specific modules for audio and metadata extraction
2. **Transcribers** (`transcribe` command) - Audio-to-text transcription using multiple providers
3. **Enrichment** (`enrich` command) - LLM-powered semantic analysis and content enhancement

Each component uses supporting infrastructure modules (see Section 6: Infrastructure Layer).

For CLI command usage and examples, see [cli-commands.md](cli-commands.md).

---

### 1. Extractors  
Platform-specific modules that handle audio and metadata extraction.

**CLI Command**: `extract`

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

Converts extracted audio into structured transcript data using multiple transcription providers.

**CLI Command**: `transcribe`

> **Note**: Transcription provider infrastructure was refactored in v0.7.5 to `pipeline/transcription/` with unified provider architecture. All "adapter" terminology was replaced with "provider" terminology. See Section 6.2 (Infrastructure Layer) for complete provider architecture details.

#### Available Transcription Providers

The transcribe command supports multiple providers (see Section 6.2 for infrastructure details):
- **LocalWhisperProvider** — Local Whisper models (privacy-first)
- **CloudOpenAIWhisperProvider** — OpenAI Whisper API
- **CloudAWSTranscribeProvider** — AWS Transcribe (enterprise-grade)

#### Transcript Processing Utilities

The following utilities in `pipeline/transcribers/` handle transcript processing:
- `normalize.py` — Transcript normalization to `TranscriptV1` format
- `validate.py` — Schema validation against `TranscriptV1`
- `persistence.py` — Transcript serialization and file output
- `schemas/transcript_v1.py` — `TranscriptV1` schema definition

---

### 3. Enrichment

AI-powered semantic enrichment for transforming transcripts into structured, semantically rich content.

**CLI Command**: `enrich`

> **Note**: LLM provider infrastructure was refactored in v0.7.5 to `pipeline/llm/` with unified provider architecture. All "agent" terminology was replaced with "provider" terminology. See Section 6.1 (Infrastructure Layer) for complete provider architecture details.

#### Available LLM Providers

The enrich command supports multiple LLM providers (see Section 6.1 for infrastructure details):
- **LocalOllamaProvider** — Local Ollama models (zero cost)
- **CloudOpenAIProvider** — OpenAI GPT models
- **CloudAnthropicProvider** — Anthropic Claude models
- **CloudAWSBedrockProvider** — AWS Bedrock (Claude and Titan)

#### `pipeline/enrichment/`

Core enrichment infrastructure for LLM-powered analysis:
- `orchestrator.py` — Coordinates enrichment workflow across providers, prompts, and validation
- `cost_estimator.py` — Pre-flight cost calculation with token counting and pricing database
- `cache.py` — File-based caching system with TTL expiration and size limits
- `chunking.py` — Automatic transcript splitting for long-form content
- `batch.py` — Batch processing with progress tracking and error handling
- `validate.py` — Schema validation and automatic repair for LLM responses
- `output.py` — Output file management with path resolution
- `errors.py` — Comprehensive error hierarchy for enrichment operations

#### `pipeline/enrichment/schemas/`

Pydantic models for enrichment output validation:
- `enrichment_v1.py` — EnrichmentV1 container with metadata
- `summary.py` — SummaryEnrichment (short/medium/long variants)
- `tag.py` — TagEnrichment (categories, keywords, entities)
- `chapter.py` — ChapterEnrichment (title, timestamps, description)
- `highlight.py` — HighlightEnrichment (quote, timestamp, importance level)

#### `pipeline/enrichment/prompts/`

YAML-based prompt engineering system:
- `loader.py` — PromptLoader for loading and caching templates
- `renderer.py` — PromptRenderer with Jinja2 templating
- `summarize.yaml`, `tag.yaml`, `chapterize.yaml`, `highlight.yaml` — Prompt templates

#### `pipeline/enrichment/presets/`

Quality and content profile configurations:
- `quality.py` — Quality presets (FAST, BALANCED, BEST)
- `content.py` — Content profiles (PODCAST, MEETING, LECTURE)

---

### 4. Configuration Management

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

### 5. Output Management

Flexible output path management introduced in v0.6.5, replacing hardcoded output directories.

#### `pipeline/output/`
- `manager.py` — OutputManager for resolving and managing output paths
- Supports absolute paths, relative paths, and directory-based output
- Automatic directory creation and unique filename generation
- Integration with configuration system for default output directories

---

### 6. Infrastructure Layer

Enterprise-grade provider architecture introduced in v0.7.5 for unified LLM and transcription service management.

**Purpose**: These infrastructure modules provide the underlying provider implementations that support the core Transcribers and Enrichment components. They are not directly invoked by users but are used internally by the CLI commands.

#### 6.1 `pipeline/llm/` — LLM Provider Infrastructure

Unified infrastructure for all LLM providers with consistent interface and configuration management (introduced in v0.7.5).

**Supports**: Enrichment component (Section 3)

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

#### 6.2 `pipeline/transcription/` — Transcription Provider Infrastructure

Unified infrastructure for all transcription providers with consistent interface and configuration management (introduced in v0.7.5).

**Supports**: Transcribers component (Section 2)

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
- **Support Role**: These modules support the core components (Transcribers and Enrichment) but are not directly user-facing

**Configuration**: Both LLM and transcription providers support flexible configuration through YAML files, environment variables, and configuration precedence rules. Pricing is also configurable for cost estimation. See [configuration-guide.md](configuration-guide.md) for complete configuration details.

---

### 7. Schema Enforcement

#### `pipeline/extractors/schema/metadata.py`

- Defines the metadata schema used by extractors (YouTube, local)
- Enforced via unit tests and integration validation
- Ensures consistent downstream consumption by agents or GUI

#### `pipeline/transcribers/schemas/transcript_v1.py`

- Defines the transcript schema used by transcription providers
- Enforced via integration tests and schema validation
- Enables structured enrichment, publishing, and archival

#### `pipeline/enrichment/schemas/`

- Defines enrichment output schemas (EnrichmentV1, Summary, Tag, Chapter, Highlight)
- Enforced via Pydantic validation with automatic repair
- Ensures consistent downstream formatting and publishing

---

### 8. Configuration & Logging

#### `pipeline/config/logging_config.py`

- Centralized logging configuration for CLI and pipeline modules  
- Supports structured logs, verbosity control, and test isolation

---

### 9. Utilities

#### `pipeline/utils/retry.py`

- Generic retry logic for transient failures (e.g., network, API)
- Used across extractors and CLI

---

## 10. Multi-Agent Protocol (Planned)

The pipeline will integrate with an MCP server to support agent-based orchestration:

- Agents will invoke extractors via CLI or direct module calls
- Metadata and audio outputs will be tagged and routed via shared schema
- Future support for GUI enrichment and real-time observability

---

## 11. Observability & Testing

- Integration tests validate CLI behavior and extractor output
- Logging is unified across all components
- Future plans include tracing and metrics for agent workflows

---

## 12. Test Coverage

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

For full folder and file layout, see [project-structure.md](project-structure.md)  


