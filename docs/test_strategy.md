# 🧪 Test Strategy

This document outlines the testing approach for the Content Pipeline project, focusing on reliability, modularity, and milestone alignment.

---

## 1. Purpose and Scope

The testing strategy ensures that all core components — extractors, CLI orchestration, and metadata schema — behave predictably across platforms and inputs. It covers unit tests, integration tests, and schema validation.

---

## 2. Test Types

- **Unit Tests**  
  Validate isolated functions and classes, especially in extractors, CLI subcommands, and schema utilities.

- **Integration Tests**  
  Simulate end-to-end flows across modular CLI, extractors, and metadata normalization.

- **Property-Based Tests**  
  Use Hypothesis to validate universal properties across CLI operations, ensuring correctness across input ranges.

- **Schema Validation**  
  Enforce field-level correctness using `pipeline/schema/metadata.py` and `TranscriptV1` schemas.

- **CLI Subcommand Tests**  
  Ensure modular CLI commands (`extract`, `transcribe`) execute correctly with shared options and help text consistency.

- **Transcript Normalization Tests**  
  Validate `TranscriptV1` schema compliance and adapter behavior across transcriber outputs.

---

## 3. Folder Layout

All tests are located in the `tests/` directory, organized by module:

---

## 🧪 `tests/` — Test Suite

This folder contains unit and integration tests for core pipeline components, organized by functionality and platform.

```text
tests/
├── assets/
│   ├── sample_audio.mp3
│   ├── sample_transcript_v1.json
│   ├── sample_transcript.txt
│   ├── sample_video_metadata.json
│   ├── sample_video.mp4
│   └── sample_whisper_raw_output.json
├── output/
├── cli/
│   ├── test_extract.py              # Extract subcommand tests
│   ├── test_transcribe.py           # Transcribe subcommand tests
│   ├── test_shared_options.py       # Shared decorator tests
│   └── test_help_texts.py           # Help text consistency tests
├── property_tests/
│   └── test_cli_properties.py       # Property-based tests for CLI refactoring
├── integration/
│   ├── test_extract_pipeline_flow.py
│   └── test_transcribe_pipeline_flow.py
├── pipeline/
│   ├── extractors/
│   │   ├── local/
│   │   │   └── test_file_audio.py
│   │   ├── youtube/
│   │   │   └── test_extractor.py
│   │   └── schema/
│   │       └── test_metadata.py
│   └── transcribers/
│       ├── adapters/
│       │   └── test_whisper_adapter.py
│       ├── schemas/
│       │   └── test_transcript_v1.py
│       ├── test_normalize.py
│       ├── test_persistence.py
│       ├── test_transcriber.py
│       └── test_validate.py
```

---

### 4. Execution

Tests are executed using `pytest`, with support for selective execution via markers defined in `pytest.ini`.

#### Marker Usage

To run only integration tests:

```bash
pytest -m "integration"
```
To exclude integration tests and run only unit tests:

```bash
pytest -m "not integration"
```

To run only property-based tests:

```bash
pytest -m "property"
```

To run only slow tests:

```bash
pytest -m "slow"
```

To exclude slow tests:

```bash
pytest -m "not slow"
```

Execution Examples
Run the full test suite:
```bash
pytest tests/
```
Run CLI-specific tests:
```bash
pytest tests/cli/
```
Run property-based tests:
```bash
pytest tests/property_tests/
```
Run a specific test function by name:
```bash
pytest tests/cli/test_transcribe.py::TestTranscribeCommand::test_transcribe_help_output
```

>**Note:**
>All tests use pytest with unittest.mock for isolation
>Integration tests reflect real CLI usage and agent orchestration

---

### 5. Mocking and Isolation

External dependencies are mocked using `unittest.mock` to ensure deterministic behavior and fast execution.

- **YouTube downloads** and file I/O are mocked to avoid network and disk dependencies  
- **Whisper transcription** is mocked in transcriber adapter tests to isolate normalization and persistence logic  
- **Metadata builders** are tested with placeholder inputs to avoid real source classification
- **CLI subcommands** use Click's CliRunner for isolated testing without subprocess calls

### 6. Property-Based Testing Strategy

Property-based tests use Hypothesis to validate universal behaviors across CLI operations:

- **Subcommand Independence**: CLI commands work without importing main_cli.py
- **Shared Decorator Equivalence**: Shared decorators behave identically to inline Click options
- **Command Routing**: Click group routes commands correctly to respective modules
- **Help Text Consistency**: Help output matches centralized constants
- **Backward Compatibility**: v0.6.0 maintains v0.5.0 CLI interface compatibility
- **Error Message Consistency**: Error conditions produce consistent, helpful messages

Each property test validates correctness across randomized input ranges, catching edge cases that unit tests might miss.

---

### 7. Testing Coverage by Milestone

The test strategy evolves with each milestone to ensure comprehensive coverage:

#### **Current Testing Focus (v0.6.0)**
- Modular CLI architecture validation
- Property-based testing for universal CLI behaviors  
- Subcommand independence and routing verification
- Shared decorator behavioral equivalence
- Help text consistency across commands
- Backward compatibility with previous CLI interface

#### **Next Testing Phase (v0.6.5)**
- Engine selection and factory pattern testing
- Configuration management and YAML parsing validation
- Environment variable integration testing
- Breaking change migration guidance verification
- Multi-engine adapter protocol conformance

*For complete milestone details, see [docs/README.md](README.md#milestone-status)*

### 8. Future Testing Plans

- Add test coverage for streaming transcription and confidence scoring  
- Validate transcript enrichment and segment filtering logic  
- Introduce extractor interface compliance tests for future platforms (TikTok, Vimeo)  
- Integrate coverage reporting and CI hooks for milestone tracking
- Multi-engine transcription quality and performance benchmarking

