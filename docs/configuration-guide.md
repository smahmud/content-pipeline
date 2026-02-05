# Configuration Guide

This guide provides comprehensive documentation for configuring the Content Pipeline, including transcription providers, LLM providers, API keys, model parameters, and pricing overrides.

---

## 📋 Overview

The Content Pipeline supports flexible configuration through multiple sources with a clear precedence hierarchy. Configuration can be provided via:

1. **CLI flags** (highest priority)
2. **Environment variables**
3. **Explicit config file** (`--config file.yaml`)
4. **Project config file** (`./.content-pipeline/config.yaml`)
5. **User config file** (`~/.content-pipeline/config.yaml`)
6. **Default values** (lowest priority)

---

## 🗂️ Configuration Files

### Project Configuration File

**Location**: `./.content-pipeline/config.yaml`

This is the primary configuration file for your project. It should be committed to version control (with sensitive values externalized to environment variables).

**Example structure**:
```yaml
# Core settings
engine: auto
output_dir: ./transcripts
log_level: info

# Transcription providers
whisper_local:
  model: base
  device: auto

whisper_api:
  api_key: ${OPENAI_API_KEY:-}
  model: whisper-1

# LLM providers
llm:
  ollama:
    base_url: http://localhost:11434
    default_model: llama2
  
  openai:
    api_key: ${OPENAI_API_KEY:-}
    default_model: gpt-4
```

### User Configuration File

**Location**: `~/.content-pipeline/config.yaml`

This file provides user-specific defaults across all projects. It's useful for:
- Personal API keys
- Preferred models
- Custom output directories

### Environment Files

The project uses `.env` files to manage environment variables for different contexts.

#### `.env.template` (Template File)

**Location**: `./.env.template` (root directory)

**Purpose**: Template showing all available environment variables with placeholder values

**Status**: Committed to version control (safe to share)

**Contents**: See the actual `.env.template` file in the repository for the complete list of variables

**Usage**: Copy this file to create your own `.env` file:
```bash
cp .env.template .env
# Then edit .env with your actual API keys
```

#### `.env` (Local Environment File)

**Location**: `./.env` (root directory)

**Purpose**: Your actual environment variables with real API keys and sensitive values

**Status**: NOT committed to version control (already in `.gitignore`)

**Security**: This file contains sensitive information - never commit it!

**Example structure**:
```bash
# Transcription API Keys
OPENAI_API_KEY=sk-proj-abc123...
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-api03-xyz789...

# Configuration
CONTENT_PIPELINE_OUTPUT_DIR=./my-transcripts
CONTENT_PIPELINE_LOG_LEVEL=debug
```

#### `.env.dev` (Development Environment File)

**Location**: `./.env.dev` (root directory)

**Purpose**: Development-specific environment variables (optional)

**Status**: Can be committed if it doesn't contain sensitive values, or kept local

**Use case**: Override settings for development without modifying `.env`

**Example structure**:
```bash
# Development overrides
CONTENT_PIPELINE_LOG_LEVEL=debug
WHISPER_LOCAL_MODEL=tiny  # Faster for testing
OLLAMA_MODEL=llama2  # Smaller model for dev
```

#### Loading Environment Files

The Content Pipeline automatically loads environment variables from `.env` files using `python-dotenv`.

**Loading order**:
1. System environment variables (highest priority)
2. `.env` file (if exists)
3. `.env.dev` file (if exists, for development)

**Manual loading** (if needed):
```python
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load .env.dev file (for development)
load_dotenv('.env.dev', override=True)
```

**Verification**:
```bash
# Check if environment variables are loaded
python -c "
import os
from dotenv import load_dotenv

load_dotenv()

print('Checking environment variables:')
print(f'OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}')
print(f'ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}')
print(f'AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not set'}')
"
```

---

## 🔑 Environment Variables

### Transcription Provider API Keys

```bash
# OpenAI Whisper API
OPENAI_API_KEY=sk-...

# AWS Transcribe
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

### LLM Provider API Keys

```bash
# OpenAI GPT
OPENAI_API_KEY=sk-...

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Ollama (local, no key needed)
OLLAMA_BASE_URL=http://localhost:11434
```

### Pricing Overrides

```bash
# Transcription pricing (cost per minute in USD)
WHISPER_API_COST_PER_MINUTE=0.006
AWS_TRANSCRIBE_COST_PER_MINUTE=0.024
```

### Model Configuration

```bash
# Whisper Local
WHISPER_LOCAL_MODEL=base

# Ollama
OLLAMA_MODEL=llama2
OLLAMA_TEMPERATURE=0.3
OLLAMA_MAX_TOKENS=4096

# OpenAI
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096

# Anthropic
ANTHROPIC_MODEL=claude-3-opus-20240229
ANTHROPIC_TEMPERATURE=0.7
ANTHROPIC_MAX_TOKENS=4096
```

### Output and Logging

```bash
# Output directory
CONTENT_PIPELINE_OUTPUT_DIR=./transcripts

# Logging level (debug, info, warning, error)
CONTENT_PIPELINE_LOG_LEVEL=info
```

---

## 🔧 Transcription Configuration

### Local Whisper Configuration

**Provider**: `LocalWhisperProvider`  
**CLI Engine**: `local-whisper`

```yaml
whisper_local:
  # Model size: tiny, base, small, medium, large
  model: base
  
  # Device: cpu, cuda, auto
  device: auto
  
  # Compute type: default, int8, int8_float16, int16, float16, float32
  compute_type: default
  
  # Timeout in seconds
  timeout: 300
  
  # Retry configuration
  retry_attempts: 3
  retry_delay: 1.0
```

**Model sizes and performance**:
- `tiny` (~39 MB) - Fastest, least accurate
- `base` (~74 MB) - Good balance (recommended for testing)
- `small` (~244 MB) - Better accuracy
- `medium` (~769 MB) - High accuracy
- `large` (~1550 MB) - Best accuracy, slowest

### OpenAI Whisper API Configuration

**Provider**: `CloudOpenAIWhisperProvider`  
**CLI Engine**: `openai-whisper`

```yaml
whisper_api:
  # API key (use environment variable)
  api_key: ${OPENAI_API_KEY:-}
  
  # Model (currently only whisper-1 available)
  model: whisper-1
  
  # Temperature (0.0 to 1.0)
  temperature: 0.0
  
  # Response format: json, text, srt, verbose_json, vtt
  response_format: json
  
  # Timeout in seconds
  timeout: 60
  
  # Retry configuration
  retry_attempts: 3
  retry_delay: 2.0
  
  # Pricing override (optional)
  cost_per_minute_usd: 0.006
```

### AWS Transcribe Configuration

**Provider**: `CloudAWSTranscribeProvider`  
**CLI Engine**: `aws-transcribe`

```yaml
aws_transcribe:
  # AWS credentials (use environment variables or IAM roles)
  access_key_id: ${AWS_ACCESS_KEY_ID:-}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY:-}
  
  # AWS region
  region: us-east-1
  
  # Language code (AWS format: en-US, es-ES, fr-FR)
  language_code: en-US
  
  # Optional S3 bucket for transcription files
  s3_bucket: null
  
  # Timeout in seconds
  timeout: 600
  
  # Retry configuration
  retry_attempts: 3
  retry_delay: 2.0
  
  # Pricing override (optional)
  cost_per_minute_usd: 0.024
```

---

## 🤖 LLM Configuration

### Local Ollama Configuration

**Provider**: `LocalOllamaProvider`  
**CLI Provider**: `ollama`

```yaml
llm:
  ollama:
    # Base URL for Ollama API
    base_url: http://localhost:11434
    
    # Default model
    default_model: llama2
    
    # Maximum tokens for responses
    max_tokens: 4096
    
    # Temperature (0.0 to 1.0)
    # Lower = more focused, Higher = more creative
    temperature: 0.3
    
    # Request timeout in seconds
    timeout: 120
```

**Popular Ollama models**:
- `llama2` - General purpose, good balance
- `llama2:13b` - Larger, better quality
- `llama2:70b` - Best quality, requires powerful hardware
- `mistral` - Fast and efficient
- `mixtral` - High quality, good for analysis

**Install models**: `ollama pull <model-name>`

### OpenAI Configuration

**Provider**: `CloudOpenAIProvider`  
**CLI Provider**: `openai`

```yaml
llm:
  openai:
    # API key (use environment variable)
    api_key: ${OPENAI_API_KEY:-}
    
    # Default model
    default_model: gpt-4
    
    # Maximum tokens for responses
    max_tokens: 4096
    
    # Temperature (0.0 to 1.0)
    temperature: 0.7
    
    # Request timeout in seconds
    timeout: 60
    
    # Pricing override (optional)
    pricing_override:
      gpt-4:
        input_per_1k: 0.03
        output_per_1k: 0.06
      gpt-4-turbo:
        input_per_1k: 0.01
        output_per_1k: 0.03
```

**Available models**:
- `gpt-4` - Best quality, highest cost
- `gpt-4-turbo` - Good balance of quality and speed
- `gpt-3.5-turbo` - Fastest, lowest cost

### Anthropic Configuration

**Provider**: `CloudAnthropicProvider`  
**CLI Provider**: `claude` or `anthropic`

```yaml
llm:
  anthropic:
    # API key (use environment variable)
    api_key: ${ANTHROPIC_API_KEY:-}
    
    # Default model
    default_model: claude-3-opus-20240229
    
    # Maximum tokens for responses
    max_tokens: 4096
    
    # Temperature (0.0 to 1.0)
    temperature: 0.7
    
    # Request timeout in seconds
    timeout: 60
```

**Available models**:
- `claude-3-opus-20240229` - Best quality, highest cost
- `claude-3-sonnet-20240229` - Good balance
- `claude-3-haiku-20240307` - Fastest, lowest cost

### AWS Bedrock Configuration

**Provider**: `CloudAWSBedrockProvider`  
**CLI Provider**: `bedrock`

```yaml
llm:
  bedrock:
    # AWS region
    region: us-east-1
    
    # AWS credentials (use environment variables or IAM roles)
    access_key_id: ${AWS_ACCESS_KEY_ID:-}
    secret_access_key: ${AWS_SECRET_ACCESS_KEY:-}
    session_token: ${AWS_SESSION_TOKEN:-}
    
    # Default model
    default_model: amazon.nova-lite-v1:0
    
    # Maximum tokens for responses
    max_tokens: 4096
    
    # Temperature (0.0 to 1.0)
    temperature: 0.7
```

**Available models**:
- `amazon.nova-lite-v1:0` - AWS native model
- `anthropic.claude-v2` - Claude v2
- `anthropic.claude-3-sonnet` - Claude 3 Sonnet
- `anthropic.claude-3-opus` - Claude 3 Opus (best quality)

---

## 💰 Pricing Configuration

### Transcription Pricing

Transcription providers use a simple per-minute pricing model.

**Configuration methods**:

1. **YAML configuration**:
```yaml
whisper_api:
  cost_per_minute_usd: 0.005  # Custom rate

aws_transcribe:
  cost_per_minute_usd: 0.020  # Custom rate
```

2. **Environment variables**:
```bash
export WHISPER_API_COST_PER_MINUTE=0.005
export AWS_TRANSCRIBE_COST_PER_MINUTE=0.020
```

**Default pricing** (as of 2024):
- OpenAI Whisper API: $0.006/minute
- AWS Transcribe: $0.024/minute
- Local Whisper: $0.00 (free)

### LLM Pricing

LLM providers use complex per-model, per-token pricing.

**Configuration method** (YAML only):
```yaml
llm:
  openai:
    pricing_override:
      gpt-4:
        input_per_1k: 0.025   # Cost per 1K input tokens
        output_per_1k: 0.050  # Cost per 1K output tokens
      gpt-4-turbo:
        input_per_1k: 0.010
        output_per_1k: 0.030
  
  anthropic:
    pricing_override:
      claude-3-opus-20240229:
        input_per_1k: 0.015
        output_per_1k: 0.075
```

**Default pricing**: Falls back to built-in pricing database if not overridden.

**Use cases for pricing overrides**:
- Enterprise volume discounts
- Regional pricing differences
- Custom pricing agreements
- Testing with different cost assumptions

---

## 🎛️ Model Parameters

### Temperature

Controls randomness in model outputs.

**Range**: 0.0 to 1.0

**Guidelines**:
- `0.0 - 0.3`: Focused, deterministic, consistent (recommended for transcription, factual tasks)
- `0.4 - 0.6`: Balanced creativity and consistency
- `0.7 - 1.0`: Creative, varied, exploratory (recommended for content generation)

**Configuration**:
```yaml
whisper_api:
  temperature: 0.0  # Transcription should be deterministic

llm:
  openai:
    temperature: 0.7  # Enrichment can be more creative
  
  ollama:
    temperature: 0.3  # Lower for factual analysis
```

### Max Tokens

Maximum number of tokens in the response.

**Guidelines**:
- Short summaries: 500-1000 tokens
- Medium content: 2000-4000 tokens
- Long-form content: 4000-8000 tokens

**Configuration**:
```yaml
llm:
  openai:
    max_tokens: 4096  # Standard limit
  
  anthropic:
    max_tokens: 8192  # Claude supports longer outputs
```

**Note**: Higher token limits increase costs and processing time.

### Timeout

Maximum time to wait for a response (in seconds).

**Guidelines**:
- Fast APIs (OpenAI, Anthropic): 60 seconds
- Slower APIs (AWS Bedrock): 120 seconds
- Local models (Ollama, Whisper): 300+ seconds

**Configuration**:
```yaml
whisper_local:
  timeout: 300  # Local processing can be slow

whisper_api:
  timeout: 60   # Cloud API is fast

llm:
  ollama:
    timeout: 120  # Local LLM can be slow
  
  openai:
    timeout: 60   # Cloud API is fast
```

### Retry Configuration

Controls retry behavior on failures.

**Configuration**:
```yaml
whisper_api:
  retry_attempts: 3  # Number of retries
  retry_delay: 2.0   # Delay between retries (seconds)

aws_transcribe:
  retry_attempts: 3
  retry_delay: 2.0
```

---

## 🔄 Environment Variable Substitution

Configuration files support environment variable substitution using the syntax:

```yaml
${VARIABLE_NAME:-default_value}
```

**Examples**:

```yaml
# Use environment variable, fallback to empty string
api_key: ${OPENAI_API_KEY:-}

# Use environment variable, fallback to default value
output_dir: ${CONTENT_PIPELINE_OUTPUT_DIR:-./transcripts}

# Use environment variable, fallback to another variable
region: ${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}
```

**Benefits**:
- Keep sensitive values out of config files
- Share config files across environments
- Override values without editing files

---

## 📝 Configuration Examples

### Privacy-Focused (Local Only)

```yaml
engine: local-whisper

whisper_local:
  model: medium
  device: auto

auto_selection:
  prefer_local: true
  fallback_enabled: false

llm:
  ollama:
    base_url: http://localhost:11434
    default_model: llama2:13b
    temperature: 0.3
```

### Quality-Focused (Cloud Services)

```yaml
engine: openai-whisper

whisper_api:
  api_key: ${OPENAI_API_KEY:-}
  temperature: 0.0

auto_selection:
  prefer_local: false

llm:
  openai:
    api_key: ${OPENAI_API_KEY:-}
    default_model: gpt-4
    temperature: 0.7
```

### Development/Testing

```yaml
engine: auto
log_level: debug

whisper_local:
  model: tiny  # Fastest for testing

llm:
  ollama:
    default_model: llama2  # Fast local model
    temperature: 0.3
```

### Enterprise (AWS)

```yaml
engine: aws-transcribe

aws_transcribe:
  region: us-east-1
  language_code: en-US

llm:
  bedrock:
    region: us-east-1
    default_model: anthropic.claude-3-sonnet
    temperature: 0.7
```

### Multi-Provider (Fallback)

```yaml
engine: auto

auto_selection:
  prefer_local: true
  fallback_enabled: true

whisper_local:
  model: base

whisper_api:
  api_key: ${OPENAI_API_KEY:-}

llm:
  ollama:
    base_url: http://localhost:11434
    default_model: llama2
  
  openai:
    api_key: ${OPENAI_API_KEY:-}
    default_model: gpt-4
```

---

## 🔍 Configuration Validation

### Verify Configuration

```bash
# Test configuration loading
python -c "
from pipeline.config.manager import ConfigurationManager
config = ConfigurationManager()
print('✅ Configuration loaded successfully')
print(f'Engine: {config.engine}')
print(f'Output dir: {config.output_dir}')
"
```

### Check Provider Availability

```bash
# Check transcription providers
python -m cli transcribe --help

# Check LLM providers
python -m cli enrich --help
```

### Validate API Keys

```bash
# Test OpenAI API key
python -c "
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ OpenAI API key is valid')
"

# Test Anthropic API key
python -c "
import os
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
print('✅ Anthropic API key is valid')
"
```

---

## 🚨 Troubleshooting

### Configuration Not Loading

**Problem**: Configuration file is not being read

**Solutions**:
1. Check file location: `./.content-pipeline/config.yaml`
2. Verify YAML syntax (use online YAML validator)
3. Check file permissions
4. Use `--config` flag to specify explicit path

### Environment Variables Not Working

**Problem**: Environment variables are not being substituted

**Solutions**:
1. Verify variable is set: `echo $VARIABLE_NAME`
2. Check syntax: `${VARIABLE_NAME:-default}`
3. Restart terminal after setting variables
4. Use `.env` file and load with `python-dotenv`

### API Key Errors

**Problem**: "Invalid API key" or "Authentication failed"

**Solutions**:
1. Verify API key is correct
2. Check for extra spaces or newlines
3. Ensure environment variable is set
4. Test API key with provider's official tools

### Provider Not Available

**Problem**: "Provider not available" error

**Solutions**:
1. Check provider is installed: `pip list | grep provider-name`
2. Verify API keys are set
3. Check network connectivity
4. For local providers (Ollama), ensure server is running

---

## 📚 Related Documentation

- [Architecture Overview](architecture.md) - System design and components
- [CLI Commands](cli-commands.md) - Command-line usage
- [Installation Guide](installation-guide.md) - Setup and dependencies
- [Infrastructure Migration Guide](infrastructure-migration-guide.md) - v0.7.5 changes

---

## 🔐 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive values
3. **Use `.env` files** for local development (already in `.gitignore`)
4. **Use IAM roles** for AWS credentials when possible
5. **Rotate API keys** regularly
6. **Use separate keys** for development and production
7. **Limit API key permissions** to minimum required scope

---

## 📊 Configuration Precedence Summary

When the same setting is defined in multiple places, the following precedence applies (highest to lowest):

1. **CLI flags** - `--engine local-whisper`
2. **Environment variables** - `WHISPER_LOCAL_MODEL=base`
3. **Explicit config file** - `--config custom.yaml`
4. **Project config** - `./.content-pipeline/config.yaml`
5. **User config** - `~/.content-pipeline/config.yaml`
6. **Default values** - Built-in defaults

This allows for flexible configuration management across different environments and use cases.
