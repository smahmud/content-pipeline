# 🧱 Project Structure

This document outlines the folder and file layout of the Content Pipeline. It reflects modularity, semantic discipline, and milestone-aligned growth across extractors, transcription, and CLI architecture.

---

## 📂 `pipeline/` — Core Modules

This folder contains the core logic for extraction, transcription, and orchestration. Each submodule is milestone-aligned and semantically scoped.

```text
pipeline/
├── extractors/              # Platform-specific logic
│   ├── base.py              # Shared interface for platform-specific extractors
│   ├── youtube/             # YouTube audio and metadata extraction
│   ├── local/               # Local file-based extraction
├── transcribers/            # Audio-to-text transcription modules
│   ├── adapters/            # Transcriber engine wrappers (v0.6.5 enhanced)
│   │   ├── base.py          # Enhanced adapter protocol with cost estimation
│   │   ├── local_whisper.py # Local Whisper adapter for privacy-first transcription
│   │   ├── openai_whisper.py # OpenAI Whisper API adapter for cloud transcription
│   │   ├── aws_transcribe.py # AWS Transcribe adapter for enterprise transcription
│   │   ├── whisper.py       # Backward compatibility adapter (deprecated)
│   │   └── auto_selector.py # Smart engine selection with intelligent fallback
│   ├── factory.py           # Engine factory pattern for adapter instantiation (v0.6.5)
│   ├── schemas/             # Transcript normalization models (e.g. transcript_v1)
├── config/                  # Configuration management (NEW in v0.6.5)
│   ├── manager.py           # ConfigurationManager for loading and merging configs
│   ├── schema.py            # Pydantic models for configuration validation
│   ├── environment.py       # Environment variable definitions
│   ├── yaml_parser.py       # YAML parsing with enhanced error reporting
│   └── pretty_printer.py    # Configuration template generation
├── output/                  # Output path management (NEW in v0.6.5)
│   └── manager.py           # OutputManager for resolving and managing output paths
├── utils/                   # Reusable helpers (e.g., retry logic)
```

## 🖥️ `cli/` — Modular CLI Architecture

Refactored in v0.6.0 into a modular, extensible CLI package. Enhanced in v0.6.5 with configuration management and engine selection:

```text
cli/
├── __init__.py              # Main CLI group and command registration
├── __main__.py              # Module execution entry point (python -m cli)
├── extract.py               # Extract subcommand implementation
├── transcribe.py            # Transcribe subcommand (v0.6.5: enhanced with engine selection)
├── shared_options.py        # Reusable option decorators (v0.6.5: added engine_option, config_option)
└── help_texts.py            # Centralized help text constants (v0.6.5: breaking change messages)
```

---

## 🧪 `tests/` — Validation Suite

- **Unit tests** for extractors, transcriber adapters, schema validators, and utility functions  
- **Integration tests** for CLI workflows (`extract`, `transcribe`) and pipeline orchestration  
- **Property-based tests** for CLI behavior validation using Hypothesis framework
- **Schema compliance** checks for metadata and transcript models (`TranscriptV1`)  
- **Persistence tests** for transcript and metadata file outputs  
- **Error handling** tests to ensure graceful failure and retry logic  
- Mirrors actual CLI invocation and source classification logic

---

## 📦 Root-Level Files

This section describes the purpose of each file located at the root of the repository.
```test
| File                     | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `README.md`              | Short project description and architecture overview                     |
| `LICENSE.md`             | License terms and usage permissions                                     |
| `changelog.md`           | Semantic version history and release notes                              |
| `Makefile`               | Developer shortcuts and task automation                                 |
| `pytest.ini`             | Pytest configuration for test discovery and behavior                    |
| `requirements.txt`       | Runtime dependencies for production use                                 |
| `requirements-dev.txt`   | Development and testing dependencies                                    |
| `requirements.lock.txt`  | Locked test environment for reproducibility                             |
| `setup.py`               | Packaging and distribution metadata                                     |
```

---

## 📘 `docs/` — Documentation Suite

This folder contains all architectural, operational, and milestone-related documentation. Each file is scoped to a specific concern to maintain clarity and avoid duplication.
```test
| File                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `README.md`            | Full project overview, key features, milestones, and licensing terms     |
| `architecture.md`      | High-level system design, agent orchestration, and milestone alignment  |
| `project_structure.md` | Explains folder layout and rationale (this file)                        |
| `cli-commands.md`      | CLI reference and development guide                                     |
| `installation-guide.md`| Setup and dependency installation guide                                 |
| `metadata_schema.md`   | Canonical schema contract and field definitions                         |
| `transcript_schema.md` | Transcript normalization model (`TranscriptV1`) and field specifications |
| `test_strategy.md`     | How unit and integration tests are structured and validated             |
```