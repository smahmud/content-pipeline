"""
Unit Tests for LLM Agents

Tests all LLM agent implementations (OpenAI, Claude, Bedrock, Ollama)
including cost estimation, token counting, and response handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.config import OpenAIConfig, AnthropicConfig, BedrockConfig, OllamaConfig
from pipeline.llm.factory import LLMProviderFactory
from pipeline.llm.errors import ConfigurationError
from pipeline.enrichment.errors import InvalidRequestError
from tests.fixtures.mock_llm_responses import (
    get_openai_response,
    get_claude_response,
    get_bedrock_response,
    get_ollama_response,
    MOCK_SUMMARY_RESPONSE
)


class TestCloudOpenAIProvider:
    """Test CloudOpenAIProvider functionality."""
    
    @pytest.fixture
    def openai_provider(self):
        """Create CloudOpenAIProvider instance."""
        config = OpenAIConfig(api_key="test_key")
        return CloudOpenAIProvider(config=config)
    
    def test_provider_initialization(self, openai_provider):
        """Test provider initializes correctly."""
        assert openai_provider.config.api_key == "test_key"
    
    def test_get_capabilities(self, openai_provider):
        """Test getting provider capabilities."""
        capabilities = openai_provider.get_capabilities()
        assert "supported_models" in capabilities
        assert "max_tokens" in capabilities
        assert len(capabilities["supported_models"]) > 0
    
    def test_get_context_window(self, openai_provider):
        """Test getting context window for different models."""
        gpt4_window = openai_provider.get_context_window("gpt-4-turbo")
        gpt35_window = openai_provider.get_context_window("gpt-3.5-turbo")
        
        assert gpt4_window > 0
        assert gpt35_window > 0
        assert gpt4_window > gpt35_window  # GPT-4 has larger window
    
    @patch('pipeline.llm.providers.cloud_openai.CloudOpenAIProvider.client', new_callable=lambda: MagicMock())
    def test_validate_requirements(self, mock_client_prop, openai_provider):
        """Test requirements validation."""
        # Mock the client.models.list() call
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        openai_provider._client = mock_client
        
        assert openai_provider.validate_requirements() is True
        
        # Should fail without API key
        config_no_key = OpenAIConfig(api_key=None)
        provider_no_key = CloudOpenAIProvider(config=config_no_key)
        assert provider_no_key.validate_requirements() is False
    
    def test_generate_success(self, openai_provider):
        """Test successful generation."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(MOCK_SUMMARY_RESPONSE)
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "test_id"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Force provider to use our mock client
        openai_provider._client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="gpt-4-turbo"
        )
        
        response = openai_provider.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.tokens_used > 0
    
    def test_estimate_cost(self, openai_provider):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="gpt-4-turbo"
        )
        
        cost = openai_provider.estimate_cost(request)
        
        assert cost > 0.0
        assert isinstance(cost, float)
    
    def test_token_counting(self, openai_provider):
        """Test token counting."""
        text = "This is a test sentence."
        tokens = openai_provider._count_tokens(text, model="gpt-4-turbo")
        
        assert tokens > 0
        assert isinstance(tokens, int)


class TestCloudAnthropicProvider:
    """Test CloudAnthropicProvider functionality."""
    
    @pytest.fixture
    def claude_provider(self):
        """Create CloudAnthropicProvider instance."""
        config = AnthropicConfig(api_key="test_key")
        return CloudAnthropicProvider(config=config)
    
    def test_provider_initialization(self, claude_provider):
        """Test provider initializes correctly."""
        capabilities = claude_provider.get_capabilities()
        assert capabilities["provider"] == "cloud-anthropic"
        assert claude_provider.config.api_key == "test_key"
    
    def test_get_capabilities(self, claude_provider):
        """Test getting provider capabilities."""
        capabilities = claude_provider.get_capabilities()
        assert "supported_models" in capabilities
        assert "claude-3-opus-20240229" in capabilities["supported_models"]
    
    def test_validate_requirements(self, claude_provider):
        """Test requirements validation."""
        # Test with valid API key
        assert claude_provider.validate_requirements() is True
        
        # Test without API key
        config_no_key = AnthropicConfig(api_key=None)
        try:
            provider_no_key = CloudAnthropicProvider(config=config_no_key)
            # Should not reach here - should raise AuthenticationError
            assert False, "Expected AuthenticationError"
        except Exception as e:
            # Expected to fail without API key
            assert "api key" in str(e).lower() or "authentication" in str(e).lower()
    
    @patch('anthropic.Anthropic')
    def test_generate_success(self, mock_anthropic, claude_provider):
        """Test successful generation."""
        # Create proper mock response object
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(MOCK_SUMMARY_RESPONSE)
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 2500
        mock_response.usage.output_tokens = 800
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Force provider to use our mock client
        claude_provider.client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3
        )
        
        response = claude_provider.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, claude_provider):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3
        )
        
        cost = claude_provider.estimate_cost(request)
        assert cost > 0.0


class TestCloudAWSBedrockProvider:
    """Test CloudAWSBedrockProvider functionality."""
    
    @pytest.fixture
    def bedrock_provider(self):
        """Create CloudAWSBedrockProvider instance."""
        config = BedrockConfig(region="us-east-1")
        return CloudAWSBedrockProvider(config=config)
    
    def test_provider_initialization(self, bedrock_provider):
        """Test provider initializes correctly."""
        capabilities = bedrock_provider.get_capabilities()
        assert capabilities["provider"] == "cloud-aws-bedrock"
        assert bedrock_provider.config.region == "us-east-1"
    
    def test_get_capabilities(self, bedrock_provider):
        """Test getting provider capabilities."""
        capabilities = bedrock_provider.get_capabilities()
        assert "supported_models" in capabilities
        assert len(capabilities["supported_models"]) > 0
    
    @patch('boto3.client')
    def test_validate_requirements(self, mock_boto, bedrock_provider):
        """Test requirements validation."""
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        
        # Should validate AWS credentials
        result = bedrock_provider.validate_requirements()
        assert isinstance(result, bool)
    
    @patch('boto3.Session')
    def test_generate_success(self, mock_session, bedrock_provider):
        """Test successful generation."""
        # Mock the boto3 session and client
        mock_client = MagicMock()
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_client
        
        # Mock the response
        mock_response_body = {
            "completion": json.dumps(MOCK_SUMMARY_RESPONSE),
            "stop_reason": "end_turn"
        }
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        # Force provider to use our mock client
        bedrock_provider.client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="anthropic.claude-v2"
        )
        
        response = bedrock_provider.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, bedrock_provider):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="anthropic.claude-v2"
        )
        
        cost = bedrock_provider.estimate_cost(request)
        assert cost >= 0.0  # Bedrock has costs


class TestLocalOllamaProvider:
    """Test LocalOllamaProvider functionality."""
    
    @pytest.fixture
    def ollama_provider(self):
        """Create LocalOllamaProvider instance."""
        config = OllamaConfig(base_url="http://localhost:11434")
        return LocalOllamaProvider(config=config)
    
    def test_provider_initialization(self, ollama_provider):
        """Test provider initializes correctly."""
        capabilities = ollama_provider.get_capabilities()
        assert capabilities["provider"] == "local-ollama"
        assert ollama_provider.config.base_url == "http://localhost:11434"
    
    def test_get_capabilities(self, ollama_provider):
        """Test getting provider capabilities."""
        capabilities = ollama_provider.get_capabilities()
        assert "supported_models" in capabilities
    
    @patch('pipeline.llm.providers.local_ollama.requests.get')
    def test_validate_requirements(self, mock_get, ollama_provider):
        """Test requirements validation (service health check)."""
        # Test service available
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        assert ollama_provider.validate_requirements() is True
        
        # Test service unavailable - use ConnectionError which is properly handled
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection error")
        assert ollama_provider.validate_requirements() is False
    
    @patch('requests.post')
    def test_generate_success(self, mock_post, ollama_provider):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = get_ollama_response(
            json.dumps(MOCK_SUMMARY_RESPONSE)
        )
        mock_post.return_value = mock_response
        
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="llama2"
        )
        
        response = ollama_provider.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, ollama_provider):
        """Test that Ollama has zero cost."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="llama2"
        )
        
        cost = ollama_provider.estimate_cost(request)
        assert cost == 0.0  # Local models are free
    
    @patch('requests.get')
    def test_list_available_models(self, mock_get, ollama_provider):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"}
            ]
        }
        mock_get.return_value = mock_response
        
        capabilities = ollama_provider.get_capabilities()
        models = capabilities["supported_models"]
        assert len(models) == 2
        assert "llama2" in models


class TestProviderFactory:
    """Test LLMProviderFactory functionality."""
    
    def test_create_openai_provider(self):
        """Test creating CloudOpenAI provider."""
        from pipeline.llm.config import LLMConfig
        
        openai_config = OpenAIConfig(api_key="test_key")
        llm_config = LLMConfig(openai=openai_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("cloud-openai")
        assert isinstance(provider, CloudOpenAIProvider)
    
    def test_create_claude_provider(self):
        """Test creating CloudAnthropic provider."""
        from pipeline.llm.config import LLMConfig
        
        anthropic_config = AnthropicConfig(api_key="test_key")
        llm_config = LLMConfig(anthropic=anthropic_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("cloud-anthropic")
        assert isinstance(provider, CloudAnthropicProvider)
    
    def test_create_bedrock_provider(self):
        """Test creating CloudAWSBedrock provider."""
        from pipeline.llm.config import LLMConfig
        
        bedrock_config = BedrockConfig(region="us-east-1")
        llm_config = LLMConfig(bedrock=bedrock_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("cloud-aws-bedrock")
        assert isinstance(provider, CloudAWSBedrockProvider)
    
    def test_create_ollama_provider(self):
        """Test creating LocalOllama provider."""
        from pipeline.llm.config import LLMConfig
        
        ollama_config = OllamaConfig(base_url="http://localhost:11434")
        llm_config = LLMConfig(ollama=ollama_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("local-ollama")
        assert isinstance(provider, LocalOllamaProvider)
    
    def test_create_invalid_provider(self):
        """Test creating provider with invalid provider."""
        from pipeline.llm.config import LLMConfig
        
        llm_config = LLMConfig()
        factory = LLMProviderFactory(config=llm_config)
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            factory.create_provider("invalid_provider")
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('pipeline.llm.providers.cloud_openai.CloudOpenAIProvider.validate_requirements')
    def test_auto_select_openai(self, mock_openai_validate):
        """Test auto-selection prefers CloudOpenAI when available."""
        from pipeline.llm.config import LLMConfig
        
        # Mock OpenAI validation to succeed
        mock_openai_validate.return_value = True
        
        openai_config = OpenAIConfig(api_key="test_key")
        llm_config = LLMConfig(openai=openai_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("auto")
        assert isinstance(provider, CloudOpenAIProvider)
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('requests.get')
    def test_auto_select_ollama_fallback(self, mock_get):
        """Test auto-selection falls back to LocalOllama."""
        from pipeline.llm.config import LLMConfig
        
        # Mock Ollama service available
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        ollama_config = OllamaConfig(base_url="http://localhost:11434")
        llm_config = LLMConfig(ollama=ollama_config)
        factory = LLMProviderFactory(config=llm_config)
        provider = factory.create_provider("auto")
        assert isinstance(provider, LocalOllamaProvider)
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('pipeline.llm.providers.local_ollama.requests.get')
    def test_auto_select_no_providers(self, mock_get):
        """Test auto-selection when no providers available."""
        from pipeline.llm.config import LLMConfig
        
        # Mock Ollama service unavailable
        mock_get.side_effect = Exception("Connection error")
        
        llm_config = LLMConfig()
        factory = LLMProviderFactory(config=llm_config)
        with pytest.raises(ConfigurationError, match="No LLM providers available"):
            factory.create_provider("auto")
    
    def test_provider_caching(self):
        """Test that providers are cached."""
        from pipeline.llm.config import LLMConfig
        
        openai_config = OpenAIConfig(api_key="test_key")
        llm_config = LLMConfig(openai=openai_config)
        factory = LLMProviderFactory(config=llm_config)
        provider1 = factory.create_provider("cloud-openai")
        provider2 = factory.create_provider("cloud-openai")
        
        # Should return same instance
        assert provider1 is provider2
    
    def test_different_configs_different_providers(self):
        """Test that different configs create different providers."""
        from pipeline.llm.config import LLMConfig
        
        openai_config1 = OpenAIConfig(api_key="key1")
        openai_config2 = OpenAIConfig(api_key="key2")
        llm_config1 = LLMConfig(openai=openai_config1)
        llm_config2 = LLMConfig(openai=openai_config2)
        factory1 = LLMProviderFactory(config=llm_config1)
        factory2 = LLMProviderFactory(config=llm_config2)
        provider1 = factory1.create_provider("cloud-openai")
        provider2 = factory2.create_provider("cloud-openai")
        
        # Should be different instances
        assert provider1 is not provider2
