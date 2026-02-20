# 🧪 Test Strategy

This document outlines the testing approach for the Content Pipeline project, focusing on reliability, modularity, and milestone alignment.

> **Quick Start**: For practical instructions on running tests, see [Testing Guide](testing-guide.md)

---

## 1. Purpose and Scope

The testing strategy ensures that all core components — extractors, CLI orchestration, and metadata schema — behave predictably across platforms and inputs. It covers unit tests, integration tests, and schema validation.

---

## 2. Test Types

- **Unit Tests**  
  Validate isolated functions and classes in extractors, CLI subcommands, transcribers, enrichment providers, and schema utilities.

- **Integration Tests**  
  Simulate end-to-end workflows across the complete pipeline (extract → transcribe → enrich).

- **Property-Based Tests**  
  Use Hypothesis to validate universal properties and invariants across CLI operations, ensuring correctness across randomized input ranges.

---

## 3. Folder Layout

All tests are located in the `tests/` directory, organized by module:

---

## 🧪 `tests/` — Test Suite

This folder contains unit and integration tests for core pipeline components, organized by functionality and platform.

```text
tests/
├── assets/                      # Test data files (not versioned)
├── output/
├── cli/
│   ├── test_extract.py              # Extract subcommand tests
│   ├── test_transcribe.py           # Transcribe subcommand tests (v0.6.5: engine selection)
│   ├── test_shared_options.py       # Shared decorator tests
│   └── test_help_texts.py           # Help text consistency tests
├── config/                           # NEW in v0.6.5: Configuration testing
│   ├── test_config_manager.py       # ConfigurationManager tests
│   ├── test_schema.py               # Configuration schema validation
│   └── test_environment_variables.py # Environment variable integration tests
├── output/                           # NEW in v0.6.5: Output management testing
│   └── test_output_manager.py       # OutputManager path resolution tests
├── property_tests/
│   ├── test_cli_properties.py       # Property-based tests for CLI refactoring
│   └── test_formatter_properties.py # NEW in v0.8.0: Formatter property tests (203 tests)
├── integration/
│   ├── test_extract_pipeline_flow.py
│   ├── test_transcribe_pipeline_flow.py
│   ├── test_formatter_orchestrator.py # NEW in v0.8.0: Formatter integration tests (25 tests)
│   └── test_format_enhancements_e2e.py # NEW in v0.8.7: Multi-source, image prompts, code samples E2E (4 tests)
├── pipeline/
│   ├── enrichment/                   # NEW in v0.7.0: Enrichment testing
│   │   ├── test_providers.py        # LLM provider tests (OpenAI, Claude, Bedrock, Ollama)
│   │   ├── test_orchestrator.py     # Enrichment workflow coordination tests
│   │   ├── test_cost_estimator.py   # Cost calculation and token counting tests
│   │   ├── test_cache.py            # Caching system tests
│   │   ├── test_chunking.py         # Long transcript handling tests
│   │   ├── test_batch.py            # Batch processing tests
│   │   ├── test_validate.py         # Schema validation and repair tests
│   │   ├── test_retry.py            # Retry logic tests
│   │   ├── test_output.py           # Output management tests
│   │   ├── test_schemas.py          # Enrichment schema tests
│   │   └── test_prompts.py          # Prompt loading and rendering tests
│   ├── extractors/
│   │   ├── local/
│   │   │   └── test_file_audio.py
│   │   ├── youtube/
│   │   │   └── test_extractor.py
│   │   └── schema/
│   │       └── test_metadata.py
│   └── transcribers/
│       ├── providers/
│       │   ├── test_local_whisper_provider.py    # v0.6.5
│       │   ├── test_openai_whisper_provider.py   # v0.6.5
│       │   ├── test_aws_transcribe_provider.py   # v0.6.5
│       │   └── test_whisper_provider.py          # Backward compatibility
│       ├── test_factory.py          # NEW in v0.6.5: Factory pattern tests
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
>Integration tests reflect real CLI usage and provider orchestration

---

### 5. Mocking and Isolation

External dependencies are mocked using `unittest.mock` to ensure deterministic behavior and fast execution.

- **YouTube downloads** and file I/O are mocked to avoid network and disk dependencies  
- **Whisper transcription** is mocked in transcriber provider tests to isolate normalization and persistence logic  
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

---

## See Also

- **[Testing Guide](testing-guide.md)** - Practical guide for running tests, markers, and workflows
- **[Architecture](architecture.md)** - System design and component overview
- **[CLI Commands](cli-commands.md)** - Command-line interface documentation

---

## Testing Coverage by Milestone

### v0.8.7 — Format Command Enhancements
- `tests/pipeline/formatters/test_source_combiner.py` — 21 tests (SourceCombiner)
- `tests/pipeline/formatters/test_image_prompts.py` — 27 tests (ImagePromptGenerator)
- `tests/pipeline/formatters/test_code_samples.py` — 34 tests (CodeSampleGenerator)
- `tests/pipeline/formatters/test_ai_video_script.py` — 43 tests (AIVideoScriptGenerator)
- `tests/pipeline/formatters/test_format_composer_extensions.py` — 18 tests (FormatComposer extensions)
- `tests/pipeline/formatters/test_video_script_schemas.py` — 17 tests (video script schemas)
- `tests/integration/test_format_enhancements_e2e.py` — 4 tests (end-to-end)
- **Total: 164 tests for v0.8.7 features**

