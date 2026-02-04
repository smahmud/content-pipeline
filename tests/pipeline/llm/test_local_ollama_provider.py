"""
Unit Tests: LocalOllamaProvider

Tests for the Local Ollama LLM provider implementation.

**Test Coverage:**
- Initialization with valid configuration
- Generate method with various inputs
- Estimate cost method (should always return 0.0)
- Validate requirements method
- Error handling for missing configuration
- Error handling for connection failures
- Context window retrieval

**Requirements Validated:**
- 10.1: Unit tests for provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.providers.base import LLMRequest, LLMResponse
from pipeline.llm.config import OllamaConfig
from pipeline.llm.errors import ProviderError, ProviderNotAvailableError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ollama_config():
    """Create a test Ollama configuration."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        default_model="llama2",
        max_tokens=4096,
        temperature=0.3,
        timeout=120
    )


@pytest.fixture
def ollama_provider(ollama_config):
    """Create a LocalOllamaProvider instance."""
    return LocalOllamaProvider(ollama_config)


@pytest.fixture
def sample_request():
    """Create a sample LLM request."""
    return LLMRequest(
        prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.7
    )


# ============================================================================
# Test: Initialization
# ============================================================================

def test_initialization_with_valid_config(ollama_config):
    """Test provider initialization with valid configuration."""
    provider = LocalOllamaProvider(ollama_config)
    
    assert provider is not None
    assert provider.config == ollama_config
    assert provider.config.base_url == "http://localhost:11434"
    assert provider.config.default_model == "llama2"


def test_initialization_stores_config(ollama_config):
    """Test that provider stores configuration object."""
    provider = LocalOllamaProvider(ollama_config)
    
    assert hasattr(provider, 'config')
    assert isinstance(provider.config, OllamaConfig)


# ============================================================================
# Test: Generate Method
# ============================================================================

@patch('requests.post')
def test_generate_with_valid_request(mock_post, ollama_provider, sample_request):
    """Test generate method with valid request."""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "The capital of France is Paris.",
        "eval_count": 50,
        "eval_duration": 1000000,
        "load_duration": 500000
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    # Generate completion
    response = ollama_provider.generate(sample_request)
    
    # Verify response
    assert isinstance(response, LLMResponse)
    assert response.content == "The capital of France is Paris."
    assert response.model_used == "llama2"
    assert response.tokens_used > 0
    assert response.cost_usd == 0.0  # Local models are free
    assert "input_tokens" in response.metadata
    assert "output_tokens" in response.metadata


@patch('requests.post')
def test_generate_uses_custom_model(mock_post, ollama_provider):
    """Test generate method uses custom model from request."""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Test response",
        "eval_count": 10
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    # Request with custom model
    request = LLMRequest(
        prompt="Test prompt",
        model="mistral",
        max_tokens=50,
        temperature=0.5
    )
    
    response = ollama_provider.generate(request)
    
    # Verify custom model was used
    assert response.model_used == "mistral"
    
    # Verify API was called with correct model
    call_args = mock_post.call_args
    assert call_args[1]['json']['model'] == "mistral"


@patch('requests.post')
def test_generate_with_metadata(mock_post, ollama_provider):
    """Test generate method passes metadata to API."""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Test response",
        "eval_count": 10
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    # Request with metadata
    request = LLMRequest(
        prompt="Test prompt",
        max_tokens=50,
        temperature=0.5,
        metadata={"top_p": 0.9, "seed": 42}
    )
    
    ollama_provider.generate(request)
    
    # Verify metadata was passed to API
    call_args = mock_post.call_args
    assert call_args[1]['json']['options']['top_p'] == 0.9
    assert call_args[1]['json']['options']['seed'] == 42


# ============================================================================
# Test: Error Handling
# ============================================================================

@patch('requests.post')
def test_generate_handles_connection_error(mock_post, ollama_provider, sample_request):
    """Test generate method handles connection errors."""
    # Mock connection error
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
    
    # Should raise ProviderNotAvailableError
    with pytest.raises(ProviderNotAvailableError) as exc_info:
        ollama_provider.generate(sample_request)
    
    assert "Cannot connect to Ollama" in str(exc_info.value)
    assert "Is Ollama running?" in str(exc_info.value)


@patch('requests.post')
def test_generate_handles_timeout(mock_post, ollama_provider, sample_request):
    """Test generate method handles timeout errors."""
    # Mock timeout error
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
        ollama_provider.generate(sample_request)
    
    assert "timed out" in str(exc_info.value)


@patch('requests.post')
def test_generate_handles_http_error(mock_post, ollama_provider, sample_request):
    """Test generate method handles HTTP errors."""
    # Mock HTTP error
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_post.return_value = mock_response
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
        ollama_provider.generate(sample_request)
    
    assert "API request failed" in str(exc_info.value)


# ============================================================================
# Test: Estimate Cost
# ============================================================================

def test_estimate_cost_returns_zero(ollama_provider, sample_request):
    """Test estimate_cost always returns 0.0 for local models."""
    cost = ollama_provider.estimate_cost(sample_request)
    
    assert cost == 0.0


def test_estimate_cost_with_different_requests(ollama_provider):
    """Test estimate_cost returns 0.0 regardless of request size."""
    # Small request
    small_request = LLMRequest(prompt="Hi", max_tokens=10, temperature=0.5)
    assert ollama_provider.estimate_cost(small_request) == 0.0
    
    # Large request
    large_request = LLMRequest(
        prompt="This is a very long prompt " * 100,
        max_tokens=4096,
        temperature=0.5
    )
    assert ollama_provider.estimate_cost(large_request) == 0.0


# ============================================================================
# Test: Validate Requirements
# ============================================================================

@patch('requests.get')
def test_validate_requirements_when_available(mock_get, ollama_provider):
    """Test validate_requirements returns True when Ollama is available."""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    result = ollama_provider.validate_requirements()
    
    assert result is True


@patch('requests.get')
def test_validate_requirements_when_unavailable(mock_get, ollama_provider):
    """Test validate_requirements returns False when Ollama is unavailable."""
    # Mock connection error
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    
    result = ollama_provider.validate_requirements()
    
    assert result is False


# ============================================================================
# Test: Get Capabilities
# ============================================================================

@patch('requests.get')
def test_get_capabilities_with_available_models(mock_get, ollama_provider):
    """Test get_capabilities returns available models."""
    # Mock successful API response with models
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
            {"name": "codellama"}
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    capabilities = ollama_provider.get_capabilities()
    
    assert capabilities["provider"] == "local-ollama"
    assert "llama2" in capabilities["supported_models"]
    assert "mistral" in capabilities["supported_models"]
    assert "codellama" in capabilities["supported_models"]
    assert capabilities["cost_per_token"] == 0.0
    assert capabilities["supports_streaming"] is True


@patch('requests.get')
def test_get_capabilities_when_unavailable(mock_get, ollama_provider):
    """Test get_capabilities returns defaults when Ollama is unavailable."""
    # Mock connection error
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    
    capabilities = ollama_provider.get_capabilities()
    
    assert capabilities["provider"] == "local-ollama"
    assert len(capabilities["supported_models"]) > 0  # Should have defaults
    assert capabilities["cost_per_token"] == 0.0


# ============================================================================
# Test: Get Context Window
# ============================================================================

def test_get_context_window_for_known_models(ollama_provider):
    """Test get_context_window returns correct values for known models."""
    assert ollama_provider.get_context_window("llama2") == 4096
    assert ollama_provider.get_context_window("llama2:7b") == 4096
    assert ollama_provider.get_context_window("mistral") == 8192
    assert ollama_provider.get_context_window("mistral:7b") == 8192


def test_get_context_window_for_unknown_model(ollama_provider):
    """Test get_context_window returns default for unknown models."""
    # Unknown model should return default
    assert ollama_provider.get_context_window("unknown-model") == 4096
    assert ollama_provider.get_context_window("custom:latest") == 4096


def test_get_context_window_handles_version_tags(ollama_provider):
    """Test get_context_window handles version tags correctly."""
    # Should match base model name
    assert ollama_provider.get_context_window("llama2:13b") == 4096
    assert ollama_provider.get_context_window("codellama:7b") == 4096


# ============================================================================
# Test: Token Counting
# ============================================================================

def test_count_tokens_approximate(ollama_provider):
    """Test approximate token counting."""
    # Test with various text lengths
    short_text = "Hello"
    assert ollama_provider._count_tokens_approximate(short_text) == 1  # 5 chars / 4
    
    medium_text = "This is a test sentence."
    expected_tokens = len(medium_text) // 4
    assert ollama_provider._count_tokens_approximate(medium_text) == expected_tokens
    
    long_text = "This is a much longer text " * 10
    expected_tokens = len(long_text) // 4
    assert ollama_provider._count_tokens_approximate(long_text) == expected_tokens


# ============================================================================
# Test: Configuration Usage
# ============================================================================

@patch('requests.post')
def test_uses_config_base_url(mock_post, ollama_config, sample_request):
    """Test that provider uses base_url from config."""
    # Create provider with custom base URL
    ollama_config.base_url = "http://custom-host:8080"
    provider = LocalOllamaProvider(ollama_config)
    
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {"response": "Test", "eval_count": 10}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    provider.generate(sample_request)
    
    # Verify custom base URL was used
    call_args = mock_post.call_args
    assert call_args[0][0].startswith("http://custom-host:8080")


@patch('requests.post')
def test_uses_config_timeout(mock_post, ollama_config, sample_request):
    """Test that provider uses timeout from config."""
    # Create provider with custom timeout
    ollama_config.timeout = 300
    provider = LocalOllamaProvider(ollama_config)
    
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {"response": "Test", "eval_count": 10}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    provider.generate(sample_request)
    
    # Verify custom timeout was used
    call_args = mock_post.call_args
    assert call_args[1]['timeout'] == 300


@patch('requests.post')
def test_uses_config_default_model(mock_post, ollama_config):
    """Test that provider uses default_model from config when not specified."""
    # Create provider with custom default model
    ollama_config.default_model = "codellama"
    provider = LocalOllamaProvider(ollama_config)
    
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {"response": "Test", "eval_count": 10}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    # Request without model specified
    request = LLMRequest(prompt="Test", max_tokens=50, temperature=0.5)
    response = provider.generate(request)
    
    # Verify default model was used
    assert response.model_used == "codellama"
