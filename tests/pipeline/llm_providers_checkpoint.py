"""
Checkpoint Test: Verify LLM Providers Work with New Configuration

This test verifies that all migrated LLM providers can be instantiated
with the new configuration system and have the expected interface.
"""

import pytest
from pipeline.llm.config import (
    OllamaConfig,
    OpenAIConfig,
    BedrockConfig,
    AnthropicConfig
)
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.providers.base import BaseLLMProvider, LLMRequest


class TestProviderInstantiation:
    """Test that all providers can be instantiated with new config."""
    
    def test_ollama_provider_instantiation(self):
        """Test LocalOllamaProvider instantiation."""
        config = OllamaConfig(
            base_url="http://localhost:11434",
            default_model="llama2",
            max_tokens=4096,
            temperature=0.3,
            timeout=120
        )
        provider = LocalOllamaProvider(config)
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.base_url == "http://localhost:11434"
        assert provider.config.default_model == "llama2"
    
    def test_openai_provider_instantiation(self):
        """Test CloudOpenAIProvider instantiation."""
        config = OpenAIConfig(
            api_key="test_key",
            default_model="gpt-4",
            max_tokens=4096,
            temperature=0.7,
            timeout=60
        )
        provider = CloudOpenAIProvider(config)
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.api_key == "test_key"
        assert provider.config.default_model == "gpt-4"
    
    def test_bedrock_provider_instantiation(self):
        """Test CloudAWSBedrockProvider instantiation."""
        config = BedrockConfig(
            region="us-east-1",
            default_model="amazon.nova-lite-v1:0",
            max_tokens=4096,
            temperature=0.7
        )
        provider = CloudAWSBedrockProvider(config)
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.region == "us-east-1"
        assert provider.config.default_model == "amazon.nova-lite-v1:0"
    
    def test_anthropic_provider_instantiation(self):
        """Test CloudAnthropicProvider instantiation."""
        config = AnthropicConfig(
            api_key="test_key",
            default_model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.7,
            timeout=60
        )
        provider = CloudAnthropicProvider(config)
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.api_key == "test_key"
        assert provider.config.default_model == "claude-3-opus-20240229"


class TestProviderInterface:
    """Test that all providers implement the required interface."""
    
    @pytest.fixture
    def ollama_provider(self):
        config = OllamaConfig()
        return LocalOllamaProvider(config)
    
    @pytest.fixture
    def openai_provider(self):
        config = OpenAIConfig(api_key="test_key")
        return CloudOpenAIProvider(config)
    
    @pytest.fixture
    def bedrock_provider(self):
        config = BedrockConfig()
        return CloudAWSBedrockProvider(config)
    
    @pytest.fixture
    def anthropic_provider(self):
        config = AnthropicConfig(api_key="test_key")
        return CloudAnthropicProvider(config)
    
    def test_providers_have_get_capabilities(self, ollama_provider, openai_provider, 
                                            bedrock_provider, anthropic_provider):
        """Test all providers have get_capabilities method."""
        for provider in [ollama_provider, openai_provider, bedrock_provider, anthropic_provider]:
            capabilities = provider.get_capabilities()
            assert isinstance(capabilities, dict)
            assert "provider" in capabilities
            assert "supported_models" in capabilities
    
    def test_providers_have_estimate_cost(self, ollama_provider, openai_provider,
                                         bedrock_provider, anthropic_provider):
        """Test all providers have estimate_cost method."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        for provider in [ollama_provider, openai_provider, bedrock_provider, anthropic_provider]:
            cost = provider.estimate_cost(request)
            assert isinstance(cost, (int, float))
            assert cost >= 0.0
    
    def test_providers_have_validate_requirements(self, ollama_provider, openai_provider,
                                                  bedrock_provider, anthropic_provider):
        """Test all providers have validate_requirements method."""
        for provider in [ollama_provider, openai_provider, bedrock_provider, anthropic_provider]:
            result = provider.validate_requirements()
            assert isinstance(result, bool)
    
    def test_providers_have_get_context_window(self, ollama_provider, openai_provider,
                                               bedrock_provider, anthropic_provider):
        """Test all providers have get_context_window method."""
        for provider in [ollama_provider, openai_provider, bedrock_provider, anthropic_provider]:
            capabilities = provider.get_capabilities()
            model = capabilities["supported_models"][0] if capabilities["supported_models"] else "default"
            window = provider.get_context_window(model)
            assert isinstance(window, int)
            assert window > 0


class TestConfigurationNoHardcoding:
    """Test that providers don't have hardcoded configuration values."""
    
    def test_ollama_uses_config_values(self):
        """Test Ollama provider uses config values, not hardcoded ones."""
        custom_config = OllamaConfig(
            base_url="http://custom:8080",
            default_model="custom-model",
            max_tokens=2000,
            temperature=0.5,
            timeout=30
        )
        provider = LocalOllamaProvider(custom_config)
        
        assert provider.config.base_url == "http://custom:8080"
        assert provider.config.default_model == "custom-model"
        assert provider.config.max_tokens == 2000
        assert provider.config.temperature == 0.5
        assert provider.config.timeout == 30
    
    def test_openai_uses_config_values(self):
        """Test OpenAI provider uses config values, not hardcoded ones."""
        custom_config = OpenAIConfig(
            api_key="custom_key",
            default_model="gpt-3.5-turbo",
            max_tokens=2000,
            temperature=0.5,
            timeout=30
        )
        provider = CloudOpenAIProvider(custom_config)
        
        assert provider.config.api_key == "custom_key"
        assert provider.config.default_model == "gpt-3.5-turbo"
        assert provider.config.max_tokens == 2000
        assert provider.config.temperature == 0.5
        assert provider.config.timeout == 30
    
    def test_bedrock_uses_config_values(self):
        """Test Bedrock provider uses config values, not hardcoded ones."""
        custom_config = BedrockConfig(
            region="eu-west-1",
            default_model="anthropic.claude-v2",
            max_tokens=2000,
            temperature=0.5
        )
        provider = CloudAWSBedrockProvider(custom_config)
        
        assert provider.config.region == "eu-west-1"
        assert provider.config.default_model == "anthropic.claude-v2"
        assert provider.config.max_tokens == 2000
        assert provider.config.temperature == 0.5
    
    def test_anthropic_uses_config_values(self):
        """Test Anthropic provider uses config values, not hardcoded ones."""
        custom_config = AnthropicConfig(
            api_key="custom_key",
            default_model="claude-3-haiku-20240307",
            max_tokens=2000,
            temperature=0.5,
            timeout=30
        )
        provider = CloudAnthropicProvider(custom_config)
        
        assert provider.config.api_key == "custom_key"
        assert provider.config.default_model == "claude-3-haiku-20240307"
        assert provider.config.max_tokens == 2000
        assert provider.config.temperature == 0.5
        assert provider.config.timeout == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
