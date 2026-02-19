# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

No unreleased changes.

## [0.8.0] - 2026-02-XX (DRAFT - DO NOT RELEASE)

### Content Formatting Release

This release introduces the `format` command for transforming enriched content into various publishing formats using a hybrid template + LLM architecture.

### Added
- **Format Command** - New CLI command for content transformation
  - 16 output format types (blog, tweet, linkedin, youtube, newsletter, seo, chapters, podcast-notes, transcript-clean, notion, obsidian, slides, meeting-minutes, quote-cards, tiktok-script, video-script)
  - Hybrid architecture: Jinja2 templates for structure + LLM enhancement for quality
  - Style profiles for consistent brand voice across outputs
  - Platform validation with character limits and constraints
  - Bundle generation for multi-format workflows
  - Batch processing with glob patterns
  - Cost estimation and control with `--max-cost` and `--dry-run`

- **Formatter Infrastructure** (`pipeline/formatters/`)
  - `FormatComposer` orchestrator for format generation workflow
  - Format-specific generators for each output type
  - Jinja2 template engine with base templates
  - LLM enhancer using `pipeline/llm/` infrastructure
  - Style profile loader and applicator
  - Input/output validation with platform constraints
  - Bundle configuration system with YAML support

- **Pre-configured Bundles**
  - `blog-launch` - Blog article with social promotion (blog, tweet, linkedin, seo)
  - `video-launch` - YouTube video with supporting content (youtube, chapters, tweet, blog)
  - `podcast` - Podcast episode package (podcast-notes, newsletter, tweet, transcript-clean)
  - `social-only` - Social media posts only (tweet, linkedin)
  - `full-repurpose` - Complete content repurposing package
  - `notes-package` - Note-taking formats (notion, obsidian, slides)

- **Style Profiles**
  - Platform-specific profiles (linkedin-professional, twitter-thread, medium-tech, etc.)
  - Tone and length customization
  - Brand voice consistency across outputs

### Changed
- Updated CLI version to 0.8.0
- Formatter uses new provider architecture (`pipeline/llm/`) introduced in v0.7.5

### Notes
- Formatter integrates with enrichment output from v0.7.0
- Uses LLM infrastructure from v0.7.5 refactoring
- All 16 output types fully implemented and tested
- See `.kiro/specs/formatter-v0.8.0/` for complete specification

## [0.7.6] - 2026-02-05

### Documentation Fixes Release

This is an **unplanned documentation-only release** to fix terminology inconsistencies introduced by the v0.7.5 infrastructure refactoring. The v0.7.5 release renamed all "adapter" terminology to "provider" and "agent" to "provider", but documentation was not fully updated.

### Changed
- Updated all documentation to use "provider" terminology consistently
- Fixed import paths in documentation examples:
  - `pipeline.enrichment.agents.factory` → `pipeline.llm.factory`
  - `pipeline.transcribers.adapters` → `pipeline.transcription.providers`
- Updated test directory structure references in testing documentation
- Fixed class name references throughout documentation:
  - `*Agent` → `*Provider`
  - `*Adapter` → `*Provider`
  - `AgentFactory` → `LLMProviderFactory`
  - `TranscriberAdapter` → `TranscriberProvider`

### Documentation Files Updated
- `docs/architecture.md` - Restructured and simplified
- `docs/project-structure.md` - Updated directory structure
- `docs/configuration-guide.md` - Created comprehensive configuration guide
- `docs/cli-commands.md` - Updated terminology and removed future planning
- `docs/README.md` - Updated milestone structure and terminology
- `docs/installation-guide.md` - Fixed verification scripts and import paths
- `docs/testing-guide.md` - Updated test structure references
- `docs/test-strategy.md` - Updated test folder layout
- `docs/metadata-schema.md` - Fixed terminology
- `docs/transcript-schema.md` - Fixed terminology
- `README.md` (root) - Updated terminology

### Notes
- No code changes - documentation-only release
- All changes follow the documentation update protocol
- Terminology now consistent with v0.7.5 codebase

## [0.7.5] - 2026-02-03

> **📖 Migration Guide**: See [docs/infrastructure-migration-guide.md](docs/infrastructure-migration-guide.md) for detailed migration instructions and code examples.

### Infrastructure Refactoring Release

This is an **unplanned technical release** focused on architectural cleanup and enterprise-grade infrastructure improvements. This refactoring was necessary to establish a solid foundation before continuing with planned feature development in v0.8.0.

### Added
- **New Infrastructure Layer** - Enterprise-grade provider architecture
  - `pipeline/llm/` - Unified LLM provider infrastructure
    - `BaseLLMProvider` protocol for consistent provider interface
    - `LocalOllamaProvider`, `CloudOpenAIProvider`, `CloudAnthropicProvider`, `CloudAWSBedrockProvider`
    - `LLMProviderFactory` with caching and auto-selection
    - `LLMConfig` with environment variable substitution and precedence
  - `pipeline/transcription/` - Unified transcription provider infrastructure
    - `TranscriberProvider` protocol for consistent provider interface
    - `LocalWhisperProvider`, `CloudOpenAIWhisperProvider`, `CloudAWSTranscribeProvider`
    - `TranscriptionProviderFactory` with caching and validation
    - `TranscriptionConfig` with environment variable substitution and precedence
  - Configuration management with YAML support and environment variable substitution
  - Comprehensive error hierarchy for both LLM and transcription domains

- **Configuration System Enhancements**
  - LLM configuration section in `.content-pipeline/config.yaml`
  - Environment variable substitution syntax: `${VAR_NAME:-default}`
  - Configuration precedence: Explicit params > Environment vars > Project config > User config > Defaults
  - Security warnings and best practices documentation
  - **Configurable pricing for all cloud providers** (allows enterprise customers to configure custom rates)
    - Transcription providers: `cost_per_minute_usd` field in `WhisperAPIConfig` and `AWSTranscribeConfig`
      - Default: $0.006/min for OpenAI Whisper API, $0.024/min for AWS Transcribe
      - Environment variables: `WHISPER_API_COST_PER_MINUTE`, `AWS_TRANSCRIBE_COST_PER_MINUTE`
      - Supports custom negotiated rates and volume discounts
    - LLM providers: `pricing_override` field for per-model pricing in `OpenAIConfig`, `BedrockConfig`, `AnthropicConfig`
      - Format: `{"model-name": {"input_per_1k": 0.01, "output_per_1k": 0.03}}`
      - Overrides default pricing database for specific models
      - Supports regional pricing differences and enterprise agreements
    - Configuration via YAML or environment variables
    - Maintains backward compatibility with default pricing values

- **Comprehensive Testing**
  - 360+ unit tests for new infrastructure
  - Property-based tests for architectural compliance
  - Integration tests for complete workflows
  - 97% test pass rate across all modules

### Changed
- **BREAKING**: Renamed all LLM classes from `*Agent` to `*Provider`
  - `BaseLLMAgent` → `BaseLLMProvider`
  - `LocalOllamaAgent` → `LocalOllamaProvider`
  - `CloudOpenAIAgent` → `CloudOpenAIProvider`
  - `CloudAnthropicAgent` → `CloudAnthropicProvider`
  - `CloudAWSBedrockAgent` → `CloudAWSBedrockProvider`
  - `AgentFactory` → `LLMProviderFactory`

- **BREAKING**: Renamed all transcription classes from `*Adapter` to `*Provider`
  - `TranscriberAdapter` → `TranscriberProvider`
  - `LocalWhisperAdapter` → `LocalWhisperProvider`
  - `OpenAIWhisperAdapter` → `CloudOpenAIWhisperProvider`
  - `AWSTranscribeAdapter` → `CloudAWSTranscribeProvider`
  - `EngineFactory` → `TranscriptionProviderFactory`

- **BREAKING**: Changed import paths for LLM infrastructure
  - `pipeline.enrichment.agents` → `pipeline.llm`
  - All enrichment, formatting, and CLI code updated

- **BREAKING**: Changed import paths for transcription infrastructure
  - `pipeline.transcribers.adapters` → `pipeline.transcription.providers`
  - All transcribers, extractors, and CLI code updated

- **BREAKING**: All providers now require configuration objects
  - No individual parameters accepted
  - Configuration objects enforce validation and type safety
  - Removed all hardcoded configuration values

- **BREAKING**: Standardized provider naming pattern
  - File naming: `{deployment}_{service}.py` (e.g., `cloud_openai.py`, `local_ollama.py`)
  - Class naming: `{Deployment}{Service}Provider` (e.g., `CloudOpenAIProvider`, `LocalOllamaProvider`)

### Removed
- **BREAKING**: Deleted old infrastructure directories
  - `pipeline/enrichment/agents/` (replaced by `pipeline/llm/`)
  - `pipeline/transcribers/adapters/` (replaced by `pipeline/transcription/providers/`)
  - All legacy adapter and agent code removed

### Migration Required
This release contains breaking changes that require code updates. See [docs/infrastructure-migration-guide.md](docs/infrastructure-migration-guide.md) for:
- Import path changes
- Class name changes
- Configuration object usage
- Before/after code examples

### Notes
- No backward compatibility maintained - development-phase refactoring
- Focus on clean architecture and enterprise-level code quality
- All tests passing (360+ unit tests, property-based tests, integration tests)
- Foundation established for v0.8.0 formatter development

## [0.7.0] - 2026-01-29

> **📖 Detailed Release Notes**: See [docs/releases/v0.7.0.md](docs/releases/v0.7.0.md) for comprehensive usage guide, configuration examples, and migration instructions.

### Added
- **LLM-Powered Enrichment System** - Complete semantic enrichment pipeline
  - Multi-provider LLM support (OpenAI, Anthropic Claude, AWS Bedrock, Local Ollama)
  - Four enrichment types: summaries, tags, chapters, highlights
  - Auto-selection with intelligent fallback between providers
  - Cost estimation and control with `--max-cost` and `--dry-run` flags
  - Quality presets (fast, balanced, best) for model selection
  - Content profiles (podcast, meeting, lecture) for domain-specific enrichment
  - Intelligent file-based caching with TTL expiration and size limits
  - Batch processing with progress tracking and error handling
  - Long transcript handling with automatic chunking and merging
  - Custom YAML prompt templates with Jinja2 rendering
  - Retry logic with exponential backoff for transient failures
  - Comprehensive error hierarchy with descriptive messages

- **LLM Agent Infrastructure**
  - `BaseLLMAgent` protocol defining unified agent interface
  - `OpenAIAgent` with tiktoken integration and pricing database
  - `ClaudeAgent` for Anthropic Claude models (Claude 2, Claude 3 Opus/Sonnet/Haiku)
  - `BedrockAgent` for AWS Bedrock (Claude and Titan models)
  - `OllamaAgent` for local models with zero cost
  - `AgentFactory` with auto-selection and credential validation

- **Enrichment Schemas**
  - `EnrichmentV1` container schema with metadata
  - `SummaryEnrichment` with short/medium/long variants
  - `TagEnrichment` with categories, keywords, entities
  - `ChapterEnrichment` with title, timestamps, description
  - `HighlightEnrichment` with quote, timestamp, importance level
  - Schema validation with automatic repair for common LLM output issues

- **Prompt Engineering System**
  - YAML-based prompt templates for all enrichment types
  - `PromptLoader` with caching and custom directory support
  - `PromptRenderer` with Jinja2 templating and context variables
  - Default prompts: summarize.yaml, tag.yaml, chapterize.yaml, highlight.yaml

- **Cost Control and Caching**
  - `CostEstimator` with provider-specific token counting
  - Pre-flight cost calculation with warning thresholds (50% of max-cost)
  - `CacheSystem` with SHA256 key generation and TTL expiration
  - Cache size limit enforcement (default 500 MB)
  - Cache hit/miss tracking in metadata

- **CLI Enrich Command**
  - `content-pipeline enrich` command with comprehensive options
  - Provider selection: `--provider` (openai, claude, bedrock, ollama, auto)
  - Model selection: `--model` (overrides quality preset)
  - Quality presets: `--quality` (fast, balanced, best)
  - Content profiles: `--preset` (podcast, meeting, lecture, custom)
  - Enrichment type flags: `--summarize`, `--tag`, `--chapterize`, `--highlight`, `--all`
  - Cost control: `--max-cost`, `--dry-run`
  - Cache control: `--no-cache`
  - Custom prompts: `--custom-prompts`
  - Batch processing: glob patterns in `--input`, `--output-dir`

- **Configuration Extensions**
  - `EnrichmentConfig` in configuration schema
  - Provider configurations (API keys, regions, endpoints)
  - Cost control settings (max_cost, warning_threshold)
  - Cache settings (enabled, ttl, max_size)
  - Custom content profile definitions

### Changed
- Updated CLI version to 0.7.0
- Extended configuration schema to support enrichment settings
- Enhanced error hierarchy with enrichment-specific errors

### Notes
- All 4 LLM providers fully implemented and tested
- 43 correctness properties defined in design document
- Comprehensive prompt templates with best practices
- Production-ready with retry logic and error handling
- See `.kiro/specs/llm-powered-enrichment/` for complete specification

## [0.6.5] - 2026-01-29

> **📖 Detailed Release Notes**: See [docs/releases/v0.6.5.md](docs/releases/v0.6.5.md) for comprehensive upgrade guide, configuration examples, and migration instructions.

### Added
- Multiple transcription engine support with explicit engine selection via `--engine` flag
- Configuration management system with YAML file support (user and project configs)
- Environment variable support for configuration and credentials
- Flexible output path management with `--output-dir` flag and configuration
- Auto-selection engine with intelligent fallback between local-whisper, openai-whisper, and aws-transcribe
- AWS Transcribe adapter for enterprise-grade transcription
- Configuration validation with Pydantic models
- Comprehensive testing guide (`docs/testing-guide.md`) with test markers and execution strategies
- Test isolation fixtures for reliable test execution
- Property-based testing for configuration management

### Changed
- **BREAKING**: Engine selection now required via `--engine` flag (no default engine)
- **BREAKING**: Engine names standardized to provider-first pattern:
  - `whisper-local` → `local-whisper`
  - `whisper-api` → `openai-whisper`
  - `aws-transcribe` unchanged
- File naming standardized to provider-first pattern:
  - `whisper_local.py` → `local_whisper.py`
  - `whisper_api.py` → `openai_whisper.py`
- Output directory no longer hardcoded to `./output/` - now configurable
- Moved configuration examples from `examples/` to `docs/examples/` for better organization
- Enhanced test strategy with slow and external test markers
- Updated all documentation to reflect v0.6.5 capabilities

### Fixed
- OpenAI Whisper adapter validation to check API key before importing openai module
- Test isolation issues with environment variables and temporary directories
- Mock paths in integration tests after file renaming

### Notes
- Backward compatibility maintained via `whisper.py` adapter wrapper
- Configuration hierarchy: CLI flags > Environment variables > Project config > User config > Defaults
- Run fast tests: `pytest -m "not slow and not external"`
- See `docs/examples/README-Configuration.md` for configuration guide

## [0.6.0] - 2026-01-26

### Added
- Modular CLI architecture with `cli/` package structure for improved maintainability
- Dual CLI entry points: `python -m cli` and `content-pipeline` console script
- Comprehensive property-based testing with 84 tests validating 12 correctness properties
- CLI documentation in `docs/cli-commands.md` and installation guide in `docs/installation-guide.md`

### Changed
- Refactored monolithic `main_cli.py` into modular subcommand architecture
- Updated `setup.py` entry point to use new CLI structure
- Enhanced CLI help system with detailed usage examples and consistent option flags
- Consolidated contributing documentation into main CLI guide

### Removed
- `main_cli.py` - replaced with modular CLI architecture
- `docs/contributing/` folder - consolidated into CLI documentation

### Notes
- No new functionality added to existing `extract` and `transcribe` commands
- All command behavior remains identical to v0.5.0
- Modular architecture establishes foundation for future command development

## [0.5.0] - 2025-11-11

### Added
- Modular CLI with `click` group and subcommands for extensible audio and transcription workflows
- `transcribe` CLI command:
  - Converts audio to text using Whisper as the transcription engine
  - Normalizes raw transcripts using the `transcript_v1` model
  - Supports `.mp3` inputs from local, cloud, or streaming sources
- Transcription powered by OpenAI Whisper in the `transcribe` command:
  - Converts `.mp3` audio to raw text using Whisper's multilingual speech recognition
  - Supports robust transcription across accents, languages, and noisy environments
  - Output is normalized using the `transcript_v1` model for punctuation, casing, and formatting consistency
- `dispatch.py` with `classify_source()` to route sources as `streaming`, `storage`, or `file_system`
- Generalized metadata schema using structural source types (`streaming`, `storage`, `file_system`)
- File-system persistence for metadata and transcript artifacts:
  - Metadata saved as `.json` alongside extracted audio
  - Transcripts saved as `.txt` after normalization
- `TranscriberAdapter` interface using `Protocol` instead of `ABC` for cleaner contracts and static validation
- Standardized file-level and function/class-level docstrings across CLI, extractors, and schema modules
- New test coverage for transcriber functionality, including Whisper transcription, normalization, and file persistence
- New test cases for:
  - CLI routing and source classification
  - Metadata generation and schema compliance
  - Transcriber adapter behavior and transcript normalization
- `help_texts.py` module to centralize CLI help text for subcommands and options

### Changed
- Refactored `generate_local_placeholder_metadata()`:
  - Renamed to `build_local_placeholder_metadata()`
  - Moved to `metadata.py` and removed `metadata_utils.py`
  - Updated to use `classify_source()` output directly for `source_type` to eliminate duplication
- Updated CLI `extract()` logic to route based on `classify_source()` and generate metadata accordingly
- Normalized docstring structure across implementation and test files:
  - Implementation modules now include filename-prefixed docstrings and scoped documentation for classes, methods, and functions
  - Test modules include file-level docstrings only, describing coverage and integration scope (e.g. `test_extract_pipeline_flow.py`)

### Removed
- `SourceType` alias from `metadata.py` (`Literal["youtube_url", "local_file"]`) in favor of centralized classification via `dispatch.classify_source()`
- First positional argument from CLI entry point, replaced by grouped subcommands (`extract`, `transcribe`, etc.)
- `--metadata-url` flag from `extract` CLI command, now handled internally via source classification and metadata builders
- `metadata_utils.py` module (functionality merged into `metadata.py`)
- Redundant type annotations and hardcoded metadata fields in placeholder builders

## [0.4.0] - 2025-10-30

### Added
- `extractors/local/` module to isolate local file extraction logic from YouTube workflows
- `BaseExtractor` abstract class to unify audio and metadata extraction interfaces across platforms
- `schema/metadata.py` to define and enforce structured metadata output across extractors
- `config/logging.py` to centralize logging configuration across CLI, extractors, and tests
- New test suite under `tests/local/` for local audio and metadata utilities
- Unit tests for extractor interface, CLI dispatch, and metadata schema compliance
- Module-level and class-level docstrings across pipeline and test files for improved traceability

### Changed
- Refactored project structure for clarity and modularity:
  - All streaming platform implementations now reside under `pipeline/extractors/youtube`
  - Local file handling moved to `pipeline/extractors/local`
  - Renamed `youtube_audio_extractor` to `pipeline` for semantic alignment
  - Introduced `pipeline/config`, `pipeline/schema`, and `pipeline/utils` for separation of concerns
  - Reorganized test suite into `tests/local/` and `tests/youtube/` for platform-specific coverage
- Migrated `retry.py` to `utils/retry.py` (no functional changes)
- Refactored from function-based extractors to class-based implementations for clearer contracts and extensibility
- Centralized logging configuration for consistent formatting across CLI, extractors, and tests
- Reorganized and expanded test coverage:
  - Reworked CLI tests to validate argument parsing and dispatch logic
  - Refactored integration tests to cover full pipeline execution with real inputs
  - Clarified unit vs integration boundaries and standardized docstring format
- Consolidated shared logic to reduce duplication and enforce DRY principles across extractors and tests

## [0.3.0] - 2025-10-28

### Added
- `extract_metadata()` implementation for YouTube extractor to return structured metadata from video source
- Metadata schema definition in `pipeline/schema/metadata.py` to enforce consistent output structure
- Unit tests for metadata extraction and schema compliance

### Changed
- Refactored YouTube extractor to support both audio and metadata extraction via unified interface
- Updated CLI to support `--metadata-url` flag and dispatch accordingly
- Expanded test coverage to validate metadata output and CLI integration

## [0.2.2] - 2025-10-27

### Added
- Integration tests to validate CLI behavior and extractor output across real input scenarios
- Logging configuration scaffold to unify log formatting and control across pipeline components

### Changed
- Hardened post-merge pipeline by centralizing logging setup in `pipeline/config/logging.py`
- Enforced consistent logging usage across CLI, extractors, and tests
- Improved error handling and retry logic in CLI and extractor modules
- Refactored test scaffolds to match actual CLI invocation and project structure

## [0.2.1] - 2025-10-25
### Added
- Retry logic with exponential backoff for audio downloads
- Logging for successful and failed operations in CLI and extractor

### Changed
- Replaced `pytubefix`, `pydub`, and `ffmpeg-python` with `yt-dlp` for more robust YouTube audio extraction
- Normalized `.mp3` output handling to prevent double extensions
- Updated `setup.py` to set version to `0.2.1` and include `yt_dlp` in `install_requires`

### Notes
- `yt-dlp` and `moviepy` require the `ffmpeg` binary to be installed and available in your system PATH
- You can download `ffmpeg` from https://ffmpeg.org/download.html

## [0.2.0] - 2025-10-24
### Added
- Support for local MP4 video files in CLI
- Unified input handling for remote and local sources
- Introduced `moviepy` for audio extraction and MP3 conversion

### Changed
- Updated `setup.py` to include `moviepy` in `install_requires`
- Set version to `0.2.0` in `setup.py`

### Notes
- Replaced `pytube` with `pytubefix` for improved YouTube video handling

## [0.1.1] - 2025-10-22
### Fixed
- Corrected `setup.py` entry point and dependency issues

## [0.1.0] - 2025-10-20
### Added
- Initial CLI for extracting audio from YouTube URLs
- MP3 conversion using `pytubefix`
