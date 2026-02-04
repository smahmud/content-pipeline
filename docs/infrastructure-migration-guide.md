# Infrastructure Migration Guide (v0.7.5)

This guide helps you migrate code from the old infrastructure (v0.7.0 and earlier) to the new infrastructure introduced in v0.7.5.

---

## Overview

Version 0.7.5 introduces a major infrastructure refactoring that establishes enterprise-grade provider architecture for LLM and transcription services. This was an **unplanned technical release** necessary to fix architectural issues before continuing with planned feature development in v0.8.0.

**Key Changes**:
- New infrastructure layer: `pipeline/llm/` and `pipeline/transcription/`
- Renamed all classes from `*Agent`/`*Adapter` to `*Provider`
- Changed import paths for better organization
- All providers now require configuration objects
- Removed all hardcoded configuration values
- Standardized naming conventions

**Breaking Changes**: This release contains breaking changes with **no backward compatibility**. All code must be updated to use the new infrastructure.

---

## Import Path Changes

### LLM Infrastructure

**Old Imports** (v0.7.0 and earlier):
```python
from pipeline.enrichment.agents.base import BaseLLMAgent
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent
from pipeline.enrichment.agents.cloud_openai_agent import CloudOpenAIAgent
from pipeline.enrichment.agents.cloud_anthropic_agent import CloudAnthropicAgent
from pipeline.enrichment.agents.cloud_aws_bedrock_agent import CloudAWSBedrockAgent
from pipeline.enrichment.agents.factory import AgentFactory
```

**New Imports** (v0.7.5):
```python
from pipeline.llm import BaseLLMProvider
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.providers.cloud_openai import CloudOpenAIProvider
from pipeline.llm.providers.cloud_anthropic import CloudAnthropicProvider
from pipeline.llm.providers.cloud_aws_bedrock import CloudAWSBedrockProvider
from pipeline.llm import LLMProviderFactory
```

### Transcription Infrastructure

**Old Imports** (v0.7.0 and earlier):
```python
from pipeline.transcribers.adapters.base import TranscriberAdapter
from pipeline.transcribers.adapters.local_whisper import LocalWhisperAdapter
from pipeline.transcribers.adapters.openai_whisper import OpenAIWhisperAdapter
from pipeline.transcribers.adapters.aws_transcribe import AWSTranscribeAdapter
from pipeline.transcribers.factory import EngineFactory
```

**New Imports** (v0.7.5):
```python
from pipeline.transcription import TranscriberProvider
from pipeline.transcription.providers.local_whisper import LocalWhisperProvider
from pipeline.transcription.providers.cloud_openai_whisper import CloudOpenAIWhisperProvider
from pipeline.transcription.providers.cloud_aws_transcribe import CloudAWSTranscribeProvider
from pipeline.transcription import TranscriptionProviderFactory
```

---

## Class Name Changes

### LLM Classes

| Old Name (v0.7.0) | New Name (v0.7.5) |
|-------------------|-------------------|
| `BaseLLMAgent` | `BaseLLMProvider` |
| `LocalOllamaAgent` | `LocalOllamaProvider` |
| `CloudOpenAIAgent` | `CloudOpenAIProvider` |
| `CloudAnthropicAgent` | `CloudAnthropicProvider` |
| `CloudAWSBedrockAgent` | `CloudAWSBedrockProvider` |
| `AgentFactory` | `LLMProviderFactory` |

### Transcription Classes

| Old Name (v0.7.0) | New Name (v0.7.5) |
|-------------------|-------------------|
| `TranscriberAdapter` | `TranscriberProvider` |
| `LocalWhisperAdapter` | `LocalWhisperProvider` |
| `OpenAIWhisperAdapter` | `CloudOpenAIWhisperProvider` |
| `AWSTranscribeAdapter` | `CloudAWSTranscribeProvider` |
| `EngineFactory` | `TranscriptionProviderFactory` |

---

## Configuration Object Usage

### LLM Configuration

**Old Way** (v0.7.0 - individual parameters):
```python
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent

# Old: Individual parameters
agent = LocalOllamaAgent(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30
)
```

**New Way** (v0.7.5 - configuration objects):
```python
from pipeline.llm import LLMConfig, LLMProviderFactory
from pipeline.llm.config import OllamaConfig

# New: Configuration object
config = LLMConfig(
    ollama=OllamaConfig(
        base_url="http://localhost:11434",
        default_model="llama2",
        timeout=30
    )
)

# Use factory to create provider
factory = LLMProviderFactory(config)
provider = factory.create_provider("local-ollama")
```

### Transcription Configuration

**Old Way** (v0.7.0 - individual parameters):
```python
from pipeline.transcribers.adapters.local_whisper import LocalWhisperAdapter

# Old: Individual parameters
adapter = LocalWhisperAdapter(
    model_name="base",
    device="cpu"
)
```

**New Way** (v0.7.5 - configuration objects):
```python
from pipeline.transcription import TranscriptionConfig, TranscriptionProviderFactory
from pipeline.transcription.config import WhisperLocalConfig

# New: Configuration object
config = TranscriptionConfig(
    whisper_local=WhisperLocalConfig(
        model="base",
        device="cpu"
    )
)

# Use factory to create provider
factory = TranscriptionProviderFactory(config)
provider = factory.create_provider("local-whisper")
```

---

## Configuration Loading from YAML

### LLM Configuration

**YAML Configuration** (`.content-pipeline/config.yaml`):
```yaml
llm:
  ollama:
    base_url: ${OLLAMA_BASE_URL:-http://localhost:11434}
    default_model: llama2
    max_tokens: 2000
    temperature: 0.7
    timeout: 30
  
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4
    max_tokens: 4000
    temperature: 0.7
    timeout: 60
```

**Loading Configuration**:
```python
from pipeline.llm import LLMConfig, LLMProviderFactory

# Load configuration from YAML
config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')

# Create factory
factory = LLMProviderFactory(config)

# Create provider (uses configuration from YAML)
provider = factory.create_provider("local-ollama")
```

### Transcription Configuration

**YAML Configuration** (`.content-pipeline/config.yaml`):
```yaml
transcription:
  whisper_local:
    model: base
    device: cpu
    compute_type: int8
    timeout: 300
  
  whisper_api:
    api_key: ${OPENAI_API_KEY}
    model: whisper-1
    temperature: 0.0
    timeout: 300
```

**Loading Configuration**:
```python
from pipeline.transcription import TranscriptionConfig, TranscriptionProviderFactory

# Load configuration from YAML
config = TranscriptionConfig.load_from_yaml('.content-pipeline/config.yaml')

# Create factory
factory = TranscriptionProviderFactory(config)

# Create provider (uses configuration from YAML)
provider = factory.create_provider("local-whisper")
```

---

## Environment Variable Substitution

The new configuration system supports environment variable substitution using the syntax: `${VAR_NAME:-default}`

**Example**:
```yaml
llm:
  openai:
    api_key: ${OPENAI_API_KEY}  # Required, no default
    default_model: ${OPENAI_MODEL:-gpt-4}  # Optional, defaults to gpt-4
```

**Configuration Precedence**:
1. Explicit parameters (passed to constructor)
2. Environment variables
3. Project configuration (`.content-pipeline/config.yaml`)
4. User configuration (`~/.content-pipeline/config.yaml`)
5. Default values

---

## Pricing Configuration

Version 0.7.5 introduces configurable pricing for accurate cost estimation with custom pricing agreements.

### Transcription Pricing

**Simple Per-Minute Pricing Model**:

**YAML Configuration** (`.content-pipeline/config.yaml`):
```yaml
whisper_api:
  api_key: ${OPENAI_API_KEY}
  model: whisper-1
  cost_per_minute_usd: 0.005  # Custom rate (default: 0.006)

aws_transcribe:
  region: us-east-1
  cost_per_minute_usd: 0.020  # Volume discount (default: 0.024)
```

**Environment Variables** (`.env` or shell):
```bash
# Override pricing via environment variables
export WHISPER_API_COST_PER_MINUTE=0.005
export AWS_TRANSCRIBE_COST_PER_MINUTE=0.020
```

**Use Cases**:
- Enterprise volume discounts
- Regional pricing differences
- Custom pricing agreements
- Testing with different cost scenarios

### LLM Pricing

**Complex Per-Model, Per-Token Pricing**:

**YAML Configuration** (`.content-pipeline/config.yaml`):
```yaml
llm:
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4
    # Override pricing for specific models
    pricing_override:
      gpt-4:
        input_per_1k: 0.025   # Custom input rate (default: 0.03)
        output_per_1k: 0.05   # Custom output rate (default: 0.06)
      gpt-3.5-turbo:
        input_per_1k: 0.0004  # Custom rate (default: 0.0005)
        output_per_1k: 0.0012 # Custom rate (default: 0.0015)
  
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229
    # Override pricing for Claude models
    pricing_override:
      claude-3-opus-20240229:
        input: 12.00   # Per 1M tokens (default: 15.00)
        output: 60.00  # Per 1M tokens (default: 75.00)
  
  bedrock:
    region: us-east-1
    default_model: anthropic.claude-3-sonnet
    # Override pricing for Bedrock models
    pricing_override:
      anthropic.claude-3-sonnet:
        input_per_1k: 0.002   # Custom rate (default: 0.003)
        output_per_1k: 0.012  # Custom rate (default: 0.015)
```

**Code Example**:
```python
from pipeline.llm import LLMConfig, LLMProviderFactory

# Load configuration with custom pricing
config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')

# Create provider
factory = LLMProviderFactory(config)
provider = factory.create_provider("cloud-openai")

# Cost estimation uses custom pricing
request = LLMRequest(
    prompt="Summarize this transcript",
    model="gpt-4",
    max_tokens=1000
)
cost = provider.estimate_cost(request)  # Uses custom pricing from config
```

**Pricing Format Differences**:
- **OpenAI/Bedrock**: `input_per_1k` and `output_per_1k` (per 1,000 tokens)
- **Anthropic**: `input` and `output` (per 1,000,000 tokens)

**Fallback Behavior**:
- If `pricing_override` is not specified, uses built-in pricing database
- If a model is not in `pricing_override`, uses built-in pricing for that model
- Built-in pricing is updated periodically to reflect provider changes

**Use Cases**:
- Enterprise agreements with negotiated rates
- Testing cost scenarios before production
- Reflecting regional pricing differences
- Accounting for volume discounts

---

## Migration Examples

### Example 1: Enrichment Orchestrator

**Before (v0.7.0)**:
```python
from pipeline.enrichment.agents.factory import AgentFactory
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator

# Old: Create agent factory
agent_factory = AgentFactory(
    ollama_base_url="http://localhost:11434",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Old: Create orchestrator
orchestrator = EnrichmentOrchestrator(agent_factory)
```

**After (v0.7.5)**:
```python
from pipeline.llm import LLMConfig, LLMProviderFactory
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator

# New: Load configuration from YAML
config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')

# New: Create provider factory
provider_factory = LLMProviderFactory(config)

# New: Create orchestrator
orchestrator = EnrichmentOrchestrator(provider_factory)
```

### Example 2: Transcription Workflow

**Before (v0.7.0)**:
```python
from pipeline.transcribers.factory import EngineFactory

# Old: Create engine factory
factory = EngineFactory()

# Old: Create adapter
adapter = factory.create_adapter("local-whisper", model_name="base")

# Old: Transcribe
result = adapter.transcribe("audio.mp3")
```

**After (v0.7.5)**:
```python
from pipeline.transcription import TranscriptionConfig, TranscriptionProviderFactory

# New: Load configuration from YAML
config = TranscriptionConfig.load_from_yaml('.content-pipeline/config.yaml')

# New: Create provider factory
factory = TranscriptionProviderFactory(config)

# New: Create provider
provider = factory.create_provider("local-whisper")

# New: Transcribe
result = provider.transcribe("audio.mp3")
```

### Example 3: CLI Command

**Before (v0.7.0)**:
```python
from pipeline.enrichment.agents.factory import AgentFactory

# Old: Create agent factory with individual parameters
agent_factory = AgentFactory(
    ollama_base_url=ollama_url,
    openai_api_key=openai_key
)

# Old: Create agent
agent = agent_factory.create_agent(provider_name)
```

**After (v0.7.5)**:
```python
from pipeline.llm import LLMConfig, LLMProviderFactory

# New: Load configuration from YAML
config = LLMConfig.load_from_yaml(config_file)

# New: Create provider factory
provider_factory = LLMProviderFactory(config)

# New: Create provider
provider = provider_factory.create_provider(provider_name)
```

---

## Provider Naming Conventions

### File Naming Pattern

**Pattern**: `{deployment}_{service}.py`

**Examples**:
- `local_ollama.py` (local deployment, ollama service)
- `cloud_openai.py` (cloud deployment, openai service)
- `cloud_anthropic.py` (cloud deployment, anthropic service)
- `cloud_aws_bedrock.py` (cloud deployment, aws bedrock service)
- `cloud_openai_whisper.py` (cloud deployment, openai whisper service)
- `cloud_aws_transcribe.py` (cloud deployment, aws transcribe service)

### Class Naming Pattern

**Pattern**: `{Deployment}{Service}Provider`

**Examples**:
- `LocalOllamaProvider` (local deployment, ollama service)
- `CloudOpenAIProvider` (cloud deployment, openai service)
- `CloudAnthropicProvider` (cloud deployment, anthropic service)
- `CloudAWSBedrockProvider` (cloud deployment, aws bedrock service)
- `CloudOpenAIWhisperProvider` (cloud deployment, openai whisper service)
- `CloudAWSTranscribeProvider` (cloud deployment, aws transcribe service)

---

## Error Handling

### LLM Errors

**New Error Hierarchy**:
```python
from pipeline.llm.errors import (
    LLMError,                    # Base exception
    ConfigurationError,          # Configuration validation failures
    ProviderError,               # Provider-specific errors
    ProviderNotAvailableError    # Provider unavailable
)

try:
    provider = factory.create_provider("local-ollama")
    result = provider.generate(request)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ProviderNotAvailableError as e:
    print(f"Provider not available: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```

### Transcription Errors

**New Error Hierarchy**:
```python
from pipeline.transcription.errors import (
    TranscriptionError,          # Base exception
    ConfigurationError,          # Configuration validation failures
    ProviderError,               # Provider-specific errors
    ProviderNotAvailableError,   # Provider unavailable
    AudioFileError,              # Audio file processing errors
    TranscriptionTimeoutError    # Timeout errors
)

try:
    provider = factory.create_provider("local-whisper")
    result = provider.transcribe("audio.mp3")
except AudioFileError as e:
    print(f"Audio file error: {e}")
except TranscriptionTimeoutError as e:
    print(f"Transcription timeout: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ProviderNotAvailableError as e:
    print(f"Provider not available: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
except TranscriptionError as e:
    print(f"Transcription error: {e}")
```

---

## Testing

### Unit Tests

**Before (v0.7.0)**:
```python
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent

def test_ollama_agent():
    agent = LocalOllamaAgent(base_url="http://localhost:11434")
    # Test agent...
```

**After (v0.7.5)**:
```python
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
from pipeline.llm.config import OllamaConfig

def test_ollama_provider():
    config = OllamaConfig(base_url="http://localhost:11434")
    provider = LocalOllamaProvider(config)
    # Test provider...
```

### Integration Tests

**Before (v0.7.0)**:
```python
from pipeline.enrichment.agents.factory import AgentFactory

def test_enrichment_workflow():
    factory = AgentFactory(ollama_base_url="http://localhost:11434")
    agent = factory.create_agent("ollama")
    # Test workflow...
```

**After (v0.7.5)**:
```python
from pipeline.llm import LLMConfig, LLMProviderFactory
from pipeline.llm.config import OllamaConfig

def test_enrichment_workflow():
    config = LLMConfig(ollama=OllamaConfig(base_url="http://localhost:11434"))
    factory = LLMProviderFactory(config)
    provider = factory.create_provider("local-ollama")
    # Test workflow...
```

---

## Common Migration Issues

### Issue 1: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'pipeline.enrichment.agents'
```

**Solution**: Update import paths to use new infrastructure:
```python
# Old
from pipeline.enrichment.agents.local_ollama_agent import LocalOllamaAgent

# New
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
```

### Issue 2: Class Not Found

**Error**:
```
AttributeError: module 'pipeline.llm' has no attribute 'LocalOllamaAgent'
```

**Solution**: Update class names to use `Provider` suffix:
```python
# Old
from pipeline.llm import LocalOllamaAgent

# New
from pipeline.llm.providers.local_ollama import LocalOllamaProvider
```

### Issue 3: Configuration Errors

**Error**:
```
TypeError: __init__() got an unexpected keyword argument 'base_url'
```

**Solution**: Use configuration objects instead of individual parameters:
```python
# Old
provider = LocalOllamaProvider(base_url="http://localhost:11434")

# New
from pipeline.llm.config import OllamaConfig
config = OllamaConfig(base_url="http://localhost:11434")
provider = LocalOllamaProvider(config)
```

### Issue 4: Factory Method Names

**Error**:
```
AttributeError: 'LLMProviderFactory' object has no attribute 'create_agent'
```

**Solution**: Update factory method names:
```python
# Old
agent = factory.create_agent("ollama")

# New
provider = factory.create_provider("local-ollama")
```

---

## Checklist

Use this checklist to ensure complete migration:

- [ ] Update all import paths from `pipeline.enrichment.agents` to `pipeline.llm`
- [ ] Update all import paths from `pipeline.transcribers.adapters` to `pipeline.transcription.providers`
- [ ] Rename all `*Agent` classes to `*Provider`
- [ ] Rename all `*Adapter` classes to `*Provider`
- [ ] Update `AgentFactory` to `LLMProviderFactory`
- [ ] Update `EngineFactory` to `TranscriptionProviderFactory`
- [ ] Replace individual parameters with configuration objects
- [ ] Update factory method calls from `create_agent()` to `create_provider()`
- [ ] Update factory method calls from `create_adapter()` to `create_provider()`
- [ ] Update provider names (e.g., "ollama" → "local-ollama")
- [ ] Update error handling to use new error classes
- [ ] Update test files with new imports and class names
- [ ] Update configuration files to use new YAML structure
- [ ] Verify all tests pass after migration

---

## Support

If you encounter issues during migration:

1. Check this migration guide for common issues
2. Review the [architecture documentation](architecture.md) for infrastructure details
3. Check the [project structure](project_structure.md) for directory layout
4. Review the [CHANGELOG](../CHANGELOG.md) for breaking changes
5. Check the test files in `tests/pipeline/llm/` and `tests/pipeline/transcription/` for examples

---

## Version History

- **v0.7.5** (2026-02-03): Infrastructure refactoring with breaking changes
- **v0.7.0** (2026-01-29): LLM-powered enrichment with agent architecture
- **v0.6.5** (2026-01-29): Enhanced transcription with adapter architecture
