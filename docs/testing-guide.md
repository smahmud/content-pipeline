# Testing Guide

## Quick Start

### Run Fast Tests (Recommended)
```bash
# Run all fast tests (excludes slow integration tests)
python -m pytest tests/ -m "not slow and not external"
```

### Run All Tests
```bash
# Run everything including slow tests (~15-20 minutes)
python -m pytest tests/ -v
```

## Test Categories

### Unit Tests
Fast tests that test individual components in isolation.
```bash
python -m pytest tests/pipeline/ -m "not integration"
```

### Property Tests
Property-based tests using Hypothesis for comprehensive validation.
```bash
python -m pytest tests/property_tests/
```

### Integration Tests
End-to-end tests that test complete workflows.
```bash
# All integration tests
python -m pytest tests/integration/

# Fast integration tests only
python -m pytest tests/integration/ -m "not slow"
```

## Test Markers

### Available Markers
- `integration`: Integration tests (slower, test complete workflows)
- `slow`: Very slow tests (5+ minutes each, usually CLI workflows)
- `external`: Tests that depend on external services (YouTube, APIs)

### Using Markers
```bash
# Run only integration tests
python -m pytest -m integration

# Run only slow tests
python -m pytest -m slow

# Exclude slow tests
python -m pytest -m "not slow"

# Exclude both slow and external tests
python -m pytest -m "not slow and not external"
```

## Test Execution Times

| Test Category | Count | Time | Marker |
|--------------|-------|------|--------|
| Adapter Tests | 5 | ~5 min | - |
| Property Tests | 201 | ~4 min | - |
| Unit Tests | 179 | ~6 min | - |
| Fast Integration | 2 | ~1 min | `integration` |
| Slow Integration | 13 | ~50 min | `slow` |
| External Tests | 1 | varies | `external` |

## Recommended Workflows

### During Development
```bash
# Run fast tests only (recommended for quick feedback)
python -m pytest tests/ -m "not slow and not external"
```

### Before Commit
```bash
# Run all tests except external dependencies
python -m pytest tests/ -m "not external"
```

### CI/CD Pipeline
```bash
# Run fast tests in CI
python -m pytest tests/ -m "not slow and not external" --cov=pipeline --cov=cli

# Run slow tests nightly
python -m pytest tests/ -m slow
```

## Test Isolation

Tests are automatically isolated using fixtures in `tests/conftest.py`:
- Temporary directories for each test
- Environment variable cleanup
- Whisper model cache cleanup
- Output directory management

## Troubleshooting

### Tests Hang
If tests hang, it's usually due to Whisper model loading. Run tests individually:
```bash
python -m pytest tests/path/to/test.py::test_name -v
```

### Tests Fail in Suite But Pass Individually
This indicates test isolation issues. Check:
1. Environment variables are being reset
2. Temporary files are being cleaned up
3. Global state is being cleared

### Slow Test Execution
Use markers to skip slow tests during development:
```bash
python -m pytest -m "not slow"
```

## Coverage

### Generate Coverage Report
```bash
# Run tests with coverage
python -m pytest tests/ --cov=pipeline --cov=cli --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Coverage Goals
- Unit tests: 90%+ coverage
- Integration tests: Validate end-to-end workflows
- Property tests: Validate invariants and edge cases

## Writing Tests

### Test File Organization
```
tests/
├── conftest.py              # Shared fixtures and test isolation
├── assets/                  # Test data files
├── pipeline/                # Unit tests mirroring pipeline/ structure
│   ├── transcribers/
│   │   ├── adapters/       # Adapter unit tests
│   │   └── test_*.py       # Component tests
│   └── ...
├── integration/             # Integration tests
│   ├── conftest.py         # Integration-specific fixtures
│   └── test_*.py           # End-to-end workflow tests
└── property_tests/          # Property-based tests
    └── test_*.py           # Hypothesis property tests
```

### Test Naming Conventions
- Unit tests: `test_<component>_<behavior>.py`
- Integration tests: `test_<workflow>_integration.py`
- Property tests: `test_<property>_properties.py`

### Adding Test Markers
```python
import pytest

@pytest.mark.integration
def test_integration_workflow():
    """Integration test."""
    pass

@pytest.mark.slow
def test_slow_workflow():
    """Slow test (5+ minutes)."""
    pass

@pytest.mark.external
def test_external_api():
    """Test that depends on external service."""
    pass
```

## CI Configuration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run fast tests
        run: |
          pip install -r requirements-dev.txt
          pytest -m "not slow and not external" --cov

  slow-tests:
    runs-on: ubuntu-latest
    # Run nightly or on main branch only
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Run slow tests
        run: |
          pip install -r requirements-dev.txt
          pytest -m slow
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [Test Strategy](docs/test_strategy.md)
- [Architecture](docs/architecture.md)
