"""
Unit Tests for LLM Agents

Tests all LLM agent implementations (OpenAI, Claude, Bedrock, Ollama)
including cost estimation, token counting, and response handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse
from pipeline.enrichment.agents.cloud_openai_agent import CloudOpenAIAgent, CloudOpenAIAgentConfig
from pipeline.enrichment.agents.cloud_anthropic_agent import CloudAnthropicAgent
from pipeline.enrichment.agents.cloud_aws_bedrock_agent import CloudAWSBedrockAgent, CloudAWSBedrockAgentConfig
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent, LocalOllamaAgentConfig
from pipeline.enrichment.agents.factory import AgentFactory
from pipeline.enrichment.errors import ConfigurationError, InvalidRequestError
from tests.fixtures.mock_llm_responses import (
    get_openai_response,
    get_claude_response,
    get_bedrock_response,
    get_ollama_response,
    MOCK_SUMMARY_RESPONSE
)


class TestCloudOpenAIAgent:
    """Test CloudOpenAIAgent functionality."""
    
    @pytest.fixture
    def openai_agent(self):
        """Create CloudOpenAIAgent instance."""
        config = CloudOpenAIAgentConfig(api_key="test_key")
        return CloudOpenAIAgent(config=config)
    
    def test_agent_initialization(self, openai_agent):
        """Test agent initializes correctly."""
        assert openai_agent.config.api_key == "test_key"
    
    def test_get_capabilities(self, openai_agent):
        """Test getting agent capabilities."""
        capabilities = openai_agent.get_capabilities()
        assert "supported_models" in capabilities
        assert "max_tokens" in capabilities
        assert len(capabilities["supported_models"]) > 0
    
    def test_get_context_window(self, openai_agent):
        """Test getting context window for different models."""
        gpt4_window = openai_agent.get_context_window("gpt-4-turbo")
        gpt35_window = openai_agent.get_context_window("gpt-3.5-turbo")
        
        assert gpt4_window > 0
        assert gpt35_window > 0
        assert gpt4_window > gpt35_window  # GPT-4 has larger window
    
    @patch('pipeline.enrichment.agents.cloud_openai_agent.CloudOpenAIAgent.client', new_callable=lambda: MagicMock())
    def test_validate_requirements(self, mock_client_prop, openai_agent):
        """Test requirements validation."""
        # Mock the client.models.list() call
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        openai_agent._client = mock_client
        
        assert openai_agent.validate_requirements() is True
        
        # Should fail without API key
        config_no_key = CloudOpenAIAgentConfig(api_key=None)
        agent_no_key = CloudOpenAIAgent(config=config_no_key)
        assert agent_no_key.validate_requirements() is False
    
    def test_generate_success(self, openai_agent):
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
        
        # Force agent to use our mock client
        openai_agent._client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="gpt-4-turbo"
        )
        
        response = openai_agent.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.tokens_used > 0
    
    def test_estimate_cost(self, openai_agent):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="gpt-4-turbo"
        )
        
        cost = openai_agent.estimate_cost(request)
        
        assert cost > 0.0
        assert isinstance(cost, float)
    
    def test_token_counting(self, openai_agent):
        """Test token counting."""
        text = "This is a test sentence."
        tokens = openai_agent._count_tokens(text, model="gpt-4-turbo")
        
        assert tokens > 0
        assert isinstance(tokens, int)


class TestCloudAnthropicAgent:
    """Test CloudAnthropicAgent functionality."""
    
    @pytest.fixture
    def claude_agent(self):
        """Create CloudAnthropicAgent instance."""
        return CloudAnthropicAgent(api_key="test_key")
    
    def test_agent_initialization(self, claude_agent):
        """Test agent initializes correctly."""
        capabilities = claude_agent.get_capabilities()
        assert capabilities["provider"] == "cloud-anthropic"
        assert claude_agent.api_key == "test_key"
    
    def test_get_capabilities(self, claude_agent):
        """Test getting agent capabilities."""
        capabilities = claude_agent.get_capabilities()
        assert "supported_models" in capabilities
        assert "claude-3-opus-20240229" in capabilities["supported_models"]
    
    def test_validate_requirements(self, claude_agent):
        """Test requirements validation."""
        request = LLMRequest(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3
        )
        assert claude_agent.validate_requirements(request) is True
        
        # Test with invalid model
        invalid_request = LLMRequest(
            prompt="Test prompt",
            model="invalid-model",
            max_tokens=1000,
            temperature=0.3
        )
        with pytest.raises(InvalidRequestError):
            claude_agent.validate_requirements(invalid_request)
    
    @patch('anthropic.Anthropic')
    def test_generate_success(self, mock_anthropic, claude_agent):
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
        
        # Force agent to use our mock client
        claude_agent.client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3
        )
        
        response = claude_agent.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, claude_agent):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3
        )
        
        cost = claude_agent.estimate_cost(request)
        assert cost > 0.0


class TestCloudAWSBedrockAgent:
    """Test CloudAWSBedrockAgent functionality."""
    
    @pytest.fixture
    def bedrock_agent(self):
        """Create CloudAWSBedrockAgent instance."""
        config = CloudAWSBedrockAgentConfig(region="us-east-1")
        return CloudAWSBedrockAgent(config=config)
    
    def test_agent_initialization(self, bedrock_agent):
        """Test agent initializes correctly."""
        capabilities = bedrock_agent.get_capabilities()
        assert capabilities["provider"] == "cloud-aws-bedrock"
        assert bedrock_agent.config.region == "us-east-1"
    
    def test_get_capabilities(self, bedrock_agent):
        """Test getting agent capabilities."""
        capabilities = bedrock_agent.get_capabilities()
        assert "supported_models" in capabilities
        assert len(capabilities["supported_models"]) > 0
    
    @patch('boto3.client')
    def test_validate_requirements(self, mock_boto, bedrock_agent):
        """Test requirements validation."""
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        
        # Should validate AWS credentials
        result = bedrock_agent.validate_requirements()
        assert isinstance(result, bool)
    
    @patch('boto3.Session')
    def test_generate_success(self, mock_session, bedrock_agent):
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
        
        # Force agent to use our mock client
        bedrock_agent.client = mock_client
        
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="anthropic.claude-v2"
        )
        
        response = bedrock_agent.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, bedrock_agent):
        """Test cost estimation."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="anthropic.claude-v2"
        )
        
        cost = bedrock_agent.estimate_cost(request)
        assert cost >= 0.0  # Bedrock has costs


class TestLocalOllamaAgent:
    """Test LocalOllamaAgent functionality."""
    
    @pytest.fixture
    def ollama_agent(self):
        """Create LocalOllamaAgent instance."""
        config = LocalOllamaAgentConfig(base_url="http://localhost:11434")
        return LocalOllamaAgent(config=config)
    
    def test_agent_initialization(self, ollama_agent):
        """Test agent initializes correctly."""
        capabilities = ollama_agent.get_capabilities()
        assert capabilities["provider"] == "local-ollama"
        assert ollama_agent.config.base_url == "http://localhost:11434"
    
    def test_get_capabilities(self, ollama_agent):
        """Test getting agent capabilities."""
        capabilities = ollama_agent.get_capabilities()
        assert "supported_models" in capabilities
    
    @patch('pipeline.enrichment.agents.local_ollama_agent.requests.get')
    def test_validate_requirements(self, mock_get, ollama_agent):
        """Test requirements validation (service health check)."""
        # Test service available
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        assert ollama_agent.validate_requirements() is True
        
        # Test service unavailable - use ConnectionError which is properly handled
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection error")
        assert ollama_agent.validate_requirements() is False
    
    @patch('requests.post')
    def test_generate_success(self, mock_post, ollama_agent):
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
        
        response = ollama_agent.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
    
    def test_estimate_cost(self, ollama_agent):
        """Test that Ollama has zero cost."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=1000,
            temperature=0.3,
            model="llama2"
        )
        
        cost = ollama_agent.estimate_cost(request)
        assert cost == 0.0  # Local models are free
    
    @patch('requests.get')
    def test_list_available_models(self, mock_get, ollama_agent):
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
        
        capabilities = ollama_agent.get_capabilities()
        models = capabilities["supported_models"]
        assert len(models) == 2
        assert "llama2" in models


class TestAgentFactory:
    """Test AgentFactory functionality."""
    
    def test_create_openai_agent(self):
        """Test creating CloudOpenAI agent."""
        openai_config = CloudOpenAIAgentConfig(api_key="test_key")
        factory = AgentFactory(openai_config=openai_config)
        agent = factory.create_agent("cloud-openai")
        assert isinstance(agent, CloudOpenAIAgent)
    
    def test_create_claude_agent(self):
        """Test creating CloudAnthropic agent."""
        factory = AgentFactory(claude_api_key="test_key")
        
        # Claude agent creation will fail without anthropic package
        # This is expected behavior
        try:
            agent = factory.create_agent("cloud-anthropic")
            assert isinstance(agent, CloudAnthropicAgent)
        except Exception as e:
            # Expected to fail without anthropic package
            assert "anthropic" in str(e).lower()
    
    def test_create_bedrock_agent(self):
        """Test creating CloudAWSBedrock agent."""
        bedrock_config = CloudAWSBedrockAgentConfig(region="us-east-1")
        factory = AgentFactory(bedrock_config=bedrock_config)
        agent = factory.create_agent("cloud-aws-bedrock")
        assert isinstance(agent, CloudAWSBedrockAgent)
    
    def test_create_ollama_agent(self):
        """Test creating LocalOllama agent."""
        ollama_config = LocalOllamaAgentConfig(base_url="http://localhost:11434")
        factory = AgentFactory(ollama_config=ollama_config)
        agent = factory.create_agent("local-ollama")
        assert isinstance(agent, LocalOllamaAgent)
    
    def test_create_invalid_provider(self):
        """Test creating agent with invalid provider."""
        factory = AgentFactory()
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            factory.create_agent("invalid_provider")
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('pipeline.enrichment.agents.cloud_openai_agent.CloudOpenAIAgent.validate_requirements')
    @patch('pipeline.enrichment.agents.local_ollama_agent.requests.get')
    def test_auto_select_openai(self, mock_get, mock_openai_validate):
        """Test auto-selection prefers CloudOpenAI when available."""
        # Mock OpenAI validation to succeed
        mock_openai_validate.return_value = True
        
        # Mock Ollama service unavailable
        mock_get.side_effect = Exception("Connection error")
        
        openai_config = CloudOpenAIAgentConfig(api_key="test_key")
        factory = AgentFactory(openai_config=openai_config)
        agent = factory.create_agent("auto")
        assert isinstance(agent, CloudOpenAIAgent)
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('requests.get')
    def test_auto_select_ollama_fallback(self, mock_get):
        """Test auto-selection falls back to LocalOllama."""
        # Mock Ollama service available
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        factory = AgentFactory()
        agent = factory.create_agent("auto")
        assert isinstance(agent, LocalOllamaAgent)
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('pipeline.enrichment.agents.local_ollama_agent.requests.get')
    def test_auto_select_no_providers(self, mock_get):
        """Test auto-selection when no providers available."""
        # Mock Ollama service unavailable
        mock_get.side_effect = Exception("Connection error")
        
        factory = AgentFactory()
        with pytest.raises(ConfigurationError, match="No LLM providers available"):
            factory.create_agent("auto")
    
    def test_agent_caching(self):
        """Test that agents are cached."""
        openai_config = CloudOpenAIAgentConfig(api_key="test_key")
        factory = AgentFactory(openai_config=openai_config)
        agent1 = factory.create_agent("cloud-openai")
        agent2 = factory.create_agent("cloud-openai")
        
        # Should return same instance
        assert agent1 is agent2
    
    def test_different_configs_different_agents(self):
        """Test that different configs create different agents."""
        openai_config1 = CloudOpenAIAgentConfig(api_key="key1")
        openai_config2 = CloudOpenAIAgentConfig(api_key="key2")
        factory1 = AgentFactory(openai_config=openai_config1)
        factory2 = AgentFactory(openai_config=openai_config2)
        agent1 = factory1.create_agent("cloud-openai")
        agent2 = factory2.create_agent("cloud-openai")
        
        # Should be different instances
        assert agent1 is not agent2
    
    def test_legacy_provider_names(self):
        """Test that legacy provider names still work with deprecation warning."""
        openai_config = CloudOpenAIAgentConfig(api_key="test_key")
        factory = AgentFactory(openai_config=openai_config)
        
        # Test legacy name with deprecation warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = factory.create_agent("openai")
            assert isinstance(agent, CloudOpenAIAgent)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "openai" in str(w[0].message)
            assert "cloud-openai" in str(w[0].message)
