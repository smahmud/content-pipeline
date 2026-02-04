"""
Unit Tests for LLMProviderFactory

Tests provider instantiation, caching, auto-selection, legacy name mapping,
and error handling for the LLM provider factory.
"""

import pytest
import warnings
from unittest.mock import Mock, patch

from pipeline.llm.factory import LLMProviderFactory, AutoSelectionConfig
from pipeline.llm.config import LLMConfig, OllamaConfig, OpenAIConfig, BedrockConfig, AnthropicConfig
from pipeline.llm.providers.base import BaseLLMProvider
from pipeline.llm.errors import ConfigurationError


class TestLLMProviderFactoryInstantiation:
    """Test provider instantiation by name."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            ollama=OllamaConfig(base_url="http://localhost:11434"),
            openai=OpenAIConfig(api_key="test_openai_key"),
            bedrock=BedrockConfig(region="us-east-1"),
            anthropic=AnthropicConfig(api_key="test_anthropic_key")
        )
    
    @pytest.fixture
    def factory(self, llm_config):
        """Create factory instance."""
        return LLMProviderFactory(llm_config)
    
    def test_factory_initialization(self, llm_config):
        """Test factory initializes with config."""
        factory = LLMProviderFactory(llm_config)
        
        assert factory.config == llm_config
        assert isinstance(factory.auto_selection, AutoSelectionConfig)
        assert factory._provider_cache == {}
    
    def test_factory_with_custom_auto_selection(self, llm_config):
        """Test factory with custom auto-selection config."""
        auto_config = AutoSelectionConfig(
            priority_order=["local-ollama", "cloud-openai"],
            fallback_enabled=False
        )
        factory = LLMProviderFactory(llm_config, auto_selection=auto_config)
        
        assert factory.auto_selection == auto_config
        assert factory.auto_selection.priority_order == ["local-ollama", "cloud-openai"]
        assert factory.auto_selection.fallback_enabled is False
    
    def test_create_ollama_provider(self, factory):
        """Test creating Ollama provider."""
        provider = factory.create_provider("local-ollama")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.base_url == "http://localhost:11434"
    
    def test_create_openai_provider(self, factory):
        """Test creating OpenAI provider."""
        provider = factory.create_provider("cloud-openai")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.api_key == "test_openai_key"
    
    def test_create_bedrock_provider(self, factory):
        """Test creating Bedrock provider."""
        provider = factory.create_provider("cloud-aws-bedrock")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.region == "us-east-1"
    
    def test_create_anthropic_provider(self, factory):
        """Test creating Anthropic provider."""
        provider = factory.create_provider("cloud-anthropic")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config.api_key == "test_anthropic_key"
    
    def test_unknown_provider_raises_error(self, factory):
        """Test that unknown provider raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("unknown-provider")
        
        assert "Unknown provider: unknown-provider" in str(exc_info.value)
        assert "Valid options:" in str(exc_info.value)


class TestProviderCaching:
    """Test provider caching prevents redundant instantiation."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            ollama=OllamaConfig(),
            openai=OpenAIConfig(api_key="test_key"),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="test_key")
        )
    
    @pytest.fixture
    def factory(self, llm_config):
        """Create factory instance."""
        return LLMProviderFactory(llm_config)
    
    def test_provider_cached_on_first_call(self, factory):
        """Test provider is cached after first instantiation."""
        assert "cloud-openai" not in factory._provider_cache
        
        provider1 = factory.create_provider("cloud-openai")
        
        assert "cloud-openai" in factory._provider_cache
        assert factory._provider_cache["cloud-openai"] is provider1
    
    def test_cached_provider_returned_on_subsequent_calls(self, factory):
        """Test same provider instance returned from cache."""
        provider1 = factory.create_provider("cloud-openai")
        provider2 = factory.create_provider("cloud-openai")
        
        # Should be the exact same instance
        assert provider1 is provider2
    
    def test_different_providers_cached_separately(self, factory):
        """Test different providers are cached independently."""
        provider1 = factory.create_provider("cloud-openai")
        provider2 = factory.create_provider("local-ollama")
        
        assert provider1 is not provider2
        assert "cloud-openai" in factory._provider_cache
        assert "local-ollama" in factory._provider_cache
    
    def test_clear_cache(self, factory):
        """Test clear_cache removes all cached providers."""
        factory.create_provider("cloud-openai")
        factory.create_provider("local-ollama")
        
        assert len(factory._provider_cache) == 2
        
        factory.clear_cache()
        
        assert len(factory._provider_cache) == 0
    
    def test_new_instance_after_clear_cache(self, factory):
        """Test new provider instance created after cache clear."""
        provider1 = factory.create_provider("cloud-openai")
        factory.clear_cache()
        provider2 = factory.create_provider("cloud-openai")
        
        # Should be different instances
        assert provider1 is not provider2


class TestAutoSelection:
    """Test auto-selection logic."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            ollama=OllamaConfig(),
            openai=OpenAIConfig(api_key="test_key"),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="test_key")
        )
    
    def test_auto_selection_returns_first_available(self, llm_config):
        """Test auto-selection returns first available provider."""
        factory = LLMProviderFactory(llm_config)
        
        with patch.object(factory, '_instantiate_provider') as mock_instantiate:
            # Create mock providers
            mock_provider = Mock(spec=BaseLLMProvider)
            mock_provider.validate_requirements.return_value = True
            mock_instantiate.return_value = mock_provider
            
            provider = factory.create_provider("auto")
            
            # Should try first provider in priority order
            assert mock_instantiate.called
            assert provider is mock_provider
    
    def test_auto_selection_skips_unavailable_providers(self, llm_config):
        """Test auto-selection skips providers that fail validation."""
        auto_config = AutoSelectionConfig(
            priority_order=["cloud-openai", "local-ollama"]
        )
        factory = LLMProviderFactory(llm_config, auto_selection=auto_config)
        
        with patch.object(factory, '_instantiate_provider') as mock_instantiate:
            # First provider fails validation, second succeeds
            mock_provider1 = Mock(spec=BaseLLMProvider)
            mock_provider1.validate_requirements.return_value = False
            
            mock_provider2 = Mock(spec=BaseLLMProvider)
            mock_provider2.validate_requirements.return_value = True
            
            mock_instantiate.side_effect = [mock_provider1, mock_provider2]
            
            provider = factory.create_provider("auto")
            
            # Should return second provider
            assert provider is mock_provider2
            assert mock_instantiate.call_count == 2
    
    def test_auto_selection_raises_error_when_none_available(self, llm_config):
        """Test auto-selection raises error when no providers available."""
        factory = LLMProviderFactory(llm_config)
        
        with patch.object(factory, '_instantiate_provider') as mock_instantiate:
            # All providers fail validation
            mock_provider = Mock(spec=BaseLLMProvider)
            mock_provider.validate_requirements.return_value = False
            mock_instantiate.return_value = mock_provider
            
            with pytest.raises(ConfigurationError) as exc_info:
                factory.create_provider("auto")
            
            assert "No LLM providers available" in str(exc_info.value)
            assert "Setup instructions:" in str(exc_info.value)
    
    def test_auto_selection_caches_selected_provider(self, llm_config):
        """Test auto-selected provider is cached."""
        factory = LLMProviderFactory(llm_config)
        
        with patch.object(factory, '_instantiate_provider') as mock_instantiate:
            mock_provider = Mock(spec=BaseLLMProvider)
            mock_provider.validate_requirements.return_value = True
            mock_instantiate.return_value = mock_provider
            
            provider = factory.create_provider("auto")
            
            # Provider should be cached
            assert len(factory._provider_cache) > 0
            assert provider in factory._provider_cache.values()


class TestErrorHandling:
    """Test error handling for missing configuration."""
    
    def test_openai_without_api_key_raises_error(self):
        """Test OpenAI provider without API key raises ConfigurationError."""
        config = LLMConfig(
            openai=OpenAIConfig(api_key=""),  # Empty API key
            ollama=OllamaConfig(),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="test_key")
        )
        factory = LLMProviderFactory(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("cloud-openai")
        
        assert "OpenAI API key not configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_anthropic_without_api_key_raises_error(self):
        """Test Anthropic provider without API key raises ConfigurationError."""
        config = LLMConfig(
            openai=OpenAIConfig(api_key="test_key"),
            ollama=OllamaConfig(),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="")  # Empty API key
        )
        factory = LLMProviderFactory(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_provider("cloud-anthropic")
        
        assert "Anthropic API key not configured" in str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    
    def test_ollama_provider_always_instantiates(self):
        """Test Ollama provider instantiates without API key."""
        config = LLMConfig(
            openai=OpenAIConfig(api_key=""),
            ollama=OllamaConfig(),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="")
        )
        factory = LLMProviderFactory(config)
        
        # Should not raise error
        provider = factory.create_provider("local-ollama")
        assert isinstance(provider, BaseLLMProvider)
    
    def test_bedrock_provider_always_instantiates(self):
        """Test Bedrock provider instantiates without explicit credentials."""
        config = LLMConfig(
            openai=OpenAIConfig(api_key=""),
            ollama=OllamaConfig(),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="")
        )
        factory = LLMProviderFactory(config)
        
        # Should not raise error (uses default credentials)
        provider = factory.create_provider("cloud-aws-bedrock")
        assert isinstance(provider, BaseLLMProvider)


class TestGetAvailableProviders:
    """Test get_available_providers method."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            ollama=OllamaConfig(),
            openai=OpenAIConfig(api_key="test_key"),
            bedrock=BedrockConfig(),
            anthropic=AnthropicConfig(api_key="test_key")
        )
    
    def test_get_available_providers_returns_list(self, llm_config):
        """Test get_available_providers returns list of provider names."""
        factory = LLMProviderFactory(llm_config)
        
        with patch.object(factory, '_instantiate_provider') as mock_instantiate:
            mock_provider = Mock(spec=BaseLLMProvider)
            mock_provider.validate_requirements.return_value = True
            mock_instantiate.return_value = mock_provider
            
            available = factory.get_available_providers()
            
            assert isinstance(available, list)
            assert len(available) > 0
    
    def test_get_available_providers_filters_unavailable(self, llm_config):
        """Test get_available_providers only returns validated providers."""
        factory = LLMProviderFactory(llm_config)
        
        def mock_instantiate_side_effect(provider):
            mock_provider = Mock(spec=BaseLLMProvider)
            # Only cloud-openai validates successfully
            mock_provider.validate_requirements.return_value = (provider == "cloud-openai")
            return mock_provider
        
        with patch.object(factory, '_instantiate_provider', side_effect=mock_instantiate_side_effect):
            available = factory.get_available_providers()
            
            assert "cloud-openai" in available
            assert len(available) == 1
    
    def test_get_available_providers_handles_errors(self, llm_config):
        """Test get_available_providers handles instantiation errors gracefully."""
        factory = LLMProviderFactory(llm_config)
        
        def mock_instantiate_side_effect(provider):
            if provider == "cloud-openai":
                raise ConfigurationError("API key missing")
            mock_provider = Mock(spec=BaseLLMProvider)
            mock_provider.validate_requirements.return_value = True
            return mock_provider
        
        with patch.object(factory, '_instantiate_provider', side_effect=mock_instantiate_side_effect):
            available = factory.get_available_providers()
            
            # Should not include cloud-openai due to error
            assert "cloud-openai" not in available
            # Should include others that succeeded
            assert len(available) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
