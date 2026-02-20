"""
Infrastructure Refactoring Integration Tests

Integration tests for complete workflows using the new infrastructure:
- Enrichment workflow with new LLM providers
- Transcription workflow with new transcription providers
- Formatting workflow with new infrastructure

These tests verify that the refactored infrastructure works end-to-end
and produces correct results.

**Validates: Requirements 10.4**
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest
from pipeline.llm.factory import LLMProviderFactory
from pipeline.llm.config import LLMConfig, OllamaConfig, OpenAIConfig
from pipeline.transcription.factory import TranscriptionProviderFactory
from pipeline.transcription.config import TranscriptionConfig, WhisperLocalConfig
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1
from pipeline.formatters.llm.enhancer import LLMEnhancer


@pytest.mark.integration
@pytest.mark.slow
class TestEnrichmentWorkflow:
    """Integration tests for complete enrichment workflow using new infrastructure."""
    
    @patch('pipeline.llm.providers.local_ollama.requests.post')
    @patch('pipeline.llm.providers.local_ollama.requests.get')
    def test_complete_enrichment_workflow_with_ollama(self, mock_get, mock_post):
        """Test complete enrichment workflow using new Ollama provider."""
        # Setup mock Ollama responses
        mock_get.return_value.status_code = 200
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': json.dumps({
                'short': 'Test summary',
                'medium': 'Test medium summary',
                'long': 'Test long summary'
            }),
            'model': 'llama2',
            'done': True
        }
        mock_post.return_value = mock_response
        
        # Create LLM config and factory using new infrastructure
        llm_config = LLMConfig(
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                default_model="llama2"
            )
        )
        
        factory = LLMProviderFactory(llm_config)
        
        # Create orchestrator
        orchestrator = EnrichmentOrchestrator(provider_factory=factory)
        
        # Create enrichment request
        request = EnrichmentRequest(
            transcript_text="This is a test transcript about machine learning.",
            language="en",
            duration=120.0,
            enrichment_types=["summary"],
            provider="local-ollama",
            model="llama2"
        )
        
        # Execute enrichment
        result = orchestrator.enrich(request)
        
        # Verify result structure
        assert isinstance(result, EnrichmentV1)
        assert result.metadata.provider == "local-ollama"
        assert result.metadata.model == "llama2"
        assert result.metadata.cost_usd == 0.0  # Ollama is free
        assert result.summary is not None
        assert result.summary.short == "Test summary"
    
    @patch('openai.OpenAI')
    def test_complete_enrichment_workflow_with_openai(self, mock_openai_class):
        """Test complete enrichment workflow using new OpenAI provider."""
        # Setup mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'short': 'OpenAI summary',
            'medium': 'OpenAI medium summary',
            'long': 'OpenAI long summary'
        })
        mock_response.model = "gpt-4"
        mock_response.usage.total_tokens = 1000
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create LLM config and factory using new infrastructure
        llm_config = LLMConfig(
            openai=OpenAIConfig(
                api_key="test_key",
                default_model="gpt-4"
            )
        )
        
        factory = LLMProviderFactory(llm_config)
        
        # Create orchestrator
        orchestrator = EnrichmentOrchestrator(provider_factory=factory)
        
        # Create enrichment request
        request = EnrichmentRequest(
            transcript_text="This is a test transcript about AI.",
            language="en",
            duration=120.0,
            enrichment_types=["summary"],
            provider="cloud-openai",
            model="gpt-4"
        )
        
        # Execute enrichment
        result = orchestrator.enrich(request)
        
        # Verify result structure
        assert isinstance(result, EnrichmentV1)
        assert result.metadata.provider == "cloud-openai"
        assert result.metadata.model == "gpt-4"
        assert result.metadata.cost_usd > 0  # OpenAI has cost
        assert result.summary is not None
        assert result.summary.short == "OpenAI summary"
    
    def test_enrichment_workflow_with_config_loading(self, tmp_path):
        """Test enrichment workflow with configuration loaded from file."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_content = """
llm:
  ollama:
    base_url: http://localhost:11434
    default_model: llama2
    max_tokens: 4096
    temperature: 0.3
"""
        config_file.write_text(config_content)
        
        # Load configuration using new infrastructure
        llm_config = LLMConfig.load_from_yaml(str(config_file))
        
        # Verify configuration was loaded correctly
        assert llm_config.ollama.base_url == "http://localhost:11434"
        assert llm_config.ollama.default_model == "llama2"
        assert llm_config.ollama.max_tokens == 4096
        assert llm_config.ollama.temperature == 0.3
        
        # Create factory with loaded config
        factory = LLMProviderFactory(llm_config)
        
        # Verify factory can create provider
        with patch('pipeline.llm.providers.local_ollama.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            provider = factory.create_provider("local-ollama")
            assert provider is not None


@pytest.mark.integration
@pytest.mark.slow
class TestTranscriptionWorkflow:
    """Integration tests for complete transcription workflow using new infrastructure."""
    
    @patch('pipeline.transcription.providers.local_whisper.whisper.load_model')
    def test_complete_transcription_workflow_with_local_whisper(self, mock_load_model, tmp_path):
        """Test complete transcription workflow using new LocalWhisperProvider."""
        # Setup mock Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'This is a test transcription.',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'This is a test transcription.'
                }
            ],
            'language': 'en'
        }
        mock_load_model.return_value = mock_model
        
        # Create transcription config using new infrastructure
        config = TranscriptionConfig(
            whisper_local=WhisperLocalConfig(
                model="base",
                device="cpu"
            )
        )
        
        # Create factory
        factory = TranscriptionProviderFactory(config)
        
        # Create provider
        provider = factory.create_provider("local-whisper")
        
        # Create test audio file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("fake audio data")
        
        # Execute transcription
        result = provider.transcribe(str(audio_file))
        
        # Verify result structure
        assert 'text' in result
        assert result['text'] == 'This is a test transcription.'
        assert 'segments' in result
        assert len(result['segments']) == 1
    
    @patch('openai.OpenAI')
    def test_complete_transcription_workflow_with_openai_whisper(self, mock_openai_class, tmp_path):
        """Test complete transcription workflow using new CloudOpenAIWhisperProvider."""
        # Setup mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "OpenAI transcription result"
        mock_response.segments = [
            {
                'start': 0.0,
                'end': 2.0,
                'text': 'OpenAI transcription result'
            }
        ]
        mock_response.language = "en"
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        # Create transcription config using new infrastructure
        config = TranscriptionConfig()
        config.whisper_api.api_key = "test_key"
        
        # Create factory
        factory = TranscriptionProviderFactory(config)
        
        # Create provider
        provider = factory.create_provider("cloud-openai-whisper")
        
        # Create test audio file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("fake audio data")
        
        # Execute transcription
        result = provider.transcribe(str(audio_file))
        
        # Verify result structure
        assert 'text' in result
        assert result['text'] == "OpenAI transcription result"
    
    def test_transcription_workflow_with_config_loading(self, tmp_path):
        """Test transcription workflow with configuration loaded from file."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_content = """
whisper_local:
  model: base
  device: cpu
  timeout: 300
whisper_api:
  api_key: test_key
  model: whisper-1
"""
        config_file.write_text(config_content)
        
        # Load configuration using new infrastructure
        config = TranscriptionConfig.load_from_yaml(str(config_file))
        
        # Verify configuration was loaded correctly
        assert config.whisper_local.model == "base"
        assert config.whisper_local.device == "cpu"
        assert config.whisper_api.api_key == "test_key"
        assert config.whisper_api.model == "whisper-1"
        
        # Create factory with loaded config
        factory = TranscriptionProviderFactory(config)
        
        # Verify factory can create providers
        with patch('pipeline.transcription.providers.local_whisper.whisper.load_model'):
            provider = factory.create_provider("local-whisper")
            assert provider is not None


@pytest.mark.integration
@pytest.mark.slow
class TestFormattingWorkflow:
    """Integration tests for complete formatting workflow using new infrastructure."""
    
    @patch('pipeline.llm.providers.local_ollama.requests.post')
    @patch('pipeline.llm.providers.local_ollama.requests.get')
    def test_complete_formatting_workflow(self, mock_get, mock_post, tmp_path):
        """Test complete formatting workflow using new LLM infrastructure."""
        # Setup mock Ollama responses
        mock_get.return_value.status_code = 200
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Enhanced formatted content',
            'model': 'llama2',
            'done': True
        }
        mock_post.return_value = mock_response
        
        # Create LLM config using new infrastructure
        llm_config = LLMConfig(
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                default_model="llama2"
            )
        )
        
        # Create factory
        factory = LLMProviderFactory(llm_config)
        
        # Create enhancer
        enhancer = LLMEnhancer(provider_factory=factory)
        
        # Create test content
        content = "This is test content that needs formatting."
        
        # Execute formatting
        result = enhancer.enhance(
            content=content,
            output_type="blog",
            provider="local-ollama",
            model="llama2"
        )
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'content')
        assert len(result.content) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Integration tests for complete end-to-end workflows."""
    
    @patch('pipeline.transcription.providers.local_whisper.whisper.load_model')
    @patch('pipeline.llm.providers.local_ollama.requests.post')
    @patch('pipeline.llm.providers.local_ollama.requests.get')
    def test_complete_pipeline_transcribe_and_enrich(
        self, mock_ollama_get, mock_ollama_post, mock_whisper_load, tmp_path
    ):
        """Test complete pipeline: transcribe audio then enrich transcript."""
        # Setup Whisper mock
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'This is a transcribed text about AI and machine learning.',
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'This is a transcribed text about AI and machine learning.'
                }
            ],
            'language': 'en'
        }
        mock_whisper_load.return_value = mock_model
        
        # Setup Ollama mock
        mock_ollama_get.return_value.status_code = 200
        
        mock_ollama_response = Mock()
        mock_ollama_response.status_code = 200
        mock_ollama_response.json.return_value = {
            'response': json.dumps({
                'short': 'AI and ML summary',
                'medium': 'Medium summary about AI',
                'long': 'Long summary about AI and machine learning'
            }),
            'model': 'llama2',
            'done': True
        }
        mock_ollama_post.return_value = mock_ollama_response
        
        # Step 1: Transcribe audio
        transcription_config = TranscriptionConfig(
            whisper_local=WhisperLocalConfig(model="base", device="cpu")
        )
        transcription_factory = TranscriptionProviderFactory(transcription_config)
        transcription_provider = transcription_factory.create_provider("local-whisper")
        
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("fake audio data")
        
        transcript_result = transcription_provider.transcribe(str(audio_file))
        
        # Verify transcription
        assert 'text' in transcript_result
        transcript_text = transcript_result['text']
        
        # Step 2: Enrich transcript
        llm_config = LLMConfig(
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                default_model="llama2"
            )
        )
        llm_factory = LLMProviderFactory(llm_config)
        orchestrator = EnrichmentOrchestrator(provider_factory=llm_factory)
        
        enrichment_request = EnrichmentRequest(
            transcript_text=transcript_text,
            language="en",
            duration=5.0,
            enrichment_types=["summary"],
            provider="local-ollama",
            model="llama2"
        )
        
        enrichment_result = orchestrator.enrich(enrichment_request)
        
        # Verify enrichment
        assert isinstance(enrichment_result, EnrichmentV1)
        assert enrichment_result.summary is not None
        assert enrichment_result.summary.short == "AI and ML summary"
        assert enrichment_result.metadata.provider == "local-ollama"
        assert enrichment_result.metadata.cost_usd == 0.0
