"""
Pytest configuration and fixtures for test isolation.
"""
import pytest
import os
import shutil
import tempfile
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_tests(tmp_path, monkeypatch):
    """
    Automatically isolate each test by:
    1. Using a temporary directory for outputs
    2. Cleaning up environment variables
    3. Resetting any global state
    """
    # Create isolated output directory
    test_output = tmp_path / "test_output"
    test_output.mkdir(exist_ok=True)
    
    # Patch common output directories to use temp
    monkeypatch.setenv("TEST_OUTPUT_DIR", str(test_output))
    
    yield
    
    # Cleanup after test
    if test_output.exists():
        shutil.rmtree(test_output, ignore_errors=True)


@pytest.fixture
def clean_whisper_cache():
    """Clean Whisper model cache between tests."""
    import gc
    yield
    # Force garbage collection to release Whisper models
    gc.collect()


@pytest.fixture(autouse=True, scope="function")
def reset_environment():
    """Reset environment variables between tests."""
    import os
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
