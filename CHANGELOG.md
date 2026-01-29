# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Planning
- **v0.7.0 LLM-Powered Enrichment** - Implementation game plan complete
  - Multi-provider LLM support (OpenAI, AWS Bedrock, Claude, Local/Ollama)
  - Semantic enrichment: summaries, tags, chapters, highlights
  - Cost control with `--max-cost` and dry-run mode
  - Quality presets and content profiles
  - Intelligent caching and batch processing
  - See [docs/milestones/milestone-v0-7-0-game-plan.md](docs/milestones/milestone-v0-7-0-game-plan.md)

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
