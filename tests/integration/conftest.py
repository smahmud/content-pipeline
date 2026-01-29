"""
Integration test specific configuration.
"""
import pytest
import os


@pytest.fixture(autouse=True)
def ensure_output_dirs():
    """Ensure output directories exist for integration tests."""
    dirs = [
        "tests/output",
        "tests/assets"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    yield
    
    # Optional: cleanup after tests
    # Note: We keep test/output for debugging but could clean it here


@pytest.fixture(autouse=True)
def slow_test_warning(request):
    """Warn about slow integration tests."""
    if 'integration' in request.keywords:
        print(f"\n⏱ Running slow integration test: {request.node.name}")
