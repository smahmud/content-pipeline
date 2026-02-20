# Contributing to Content Pipeline

Thanks for your interest in contributing. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/smahmud/content-pipeline.git
cd content-pipeline
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Running Tests

```bash
# Unit and CLI tests
python -m pytest tests/pipeline/ tests/cli/ tests/mcp_server/ tests/api/ -v

# Specific test file
python -m pytest tests/pipeline/validation/test_engine.py -v
```

## Coding Standards

- Follow PEP 8 with max line length of 120
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep imports organized (stdlib, third-party, local)

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Write tests for new functionality
3. Ensure all tests pass before submitting
4. Update documentation if adding new commands or features
5. Submit PR with a clear description of changes

## Schema Policy

TranscriptV1, EnrichmentV1, and FormatV1 schemas are frozen as of v1.0.0. New fields may be added (backward compatible) but existing fields must not be removed or have their types changed.

## Questions?

Open an issue on GitHub for questions or discussion.
