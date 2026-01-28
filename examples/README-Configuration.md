# Content Pipeline Configuration Guide

This directory contains example configuration files for the Content Pipeline v0.6.5 enhanced transcription system.

## Configuration Hierarchy

The Content Pipeline uses a hierarchical configuration system with the following precedence (highest to lowest):

1. **CLI Flags** - Command-line options like `--engine`, `--output-dir`
2. **Environment Variables** - `CONTENT_PIPELINE_*` and service-specific variables
3. **Project Config** - `./.content-pipeline/config.yaml` in your project directory
4. **User Config** - `~/.content-pipeline/config.yaml` in your home directory
5. **System Defaults** - Built-in default values

## Configuration Files

### Project Configuration
- **Location**: `./.content-pipeline/config.yaml` (in your project root)
- **Purpose**: Project-specific settings that apply to all team members
- **Example**: `project-config-minimal.yaml`
- **Use Case**: Set project-wide engine preferences, output directories, language settings

### User Configuration
- **Location**: `~/.content-pipeline/config.yaml` (in your home directory)
- **Purpose**: Personal preferences that apply across all projects
- **Example**: `user-config-template.yaml`
- **Use Case**: Set your preferred engine, API keys, logging preferences

### Production Configuration
- **Location**: Deployed as project or user config in production environment
- **Purpose**: Optimized settings for production deployment
- **Example**: `production-config.yaml`
- **Use Case**: Enterprise deployments with monitoring, security, reliability features

## Quick Start

### 1. Set Up User Configuration
```bash
# Create user config directory
mkdir -p ~/.content-pipeline

# Copy and customize user template
cp examples/user-config-template.yaml ~/.content-pipeline/config.yaml

# Edit with your preferences
nano ~/.content-pipeline/config.yaml
```

### 2. Set Up Project Configuration
```bash
# Create project config directory
mkdir -p ./.content-pipeline

# Copy and customize project template
cp examples/project-config-minimal.yaml ./.content-pipeline/config.yaml

# Edit with project-specific settings
nano ./.content-pipeline/config.yaml
```

### 3. Set Environment Variables
```bash
# OpenAI API key (if using whisper-api)
export OPENAI_API_KEY="your-api-key-here"

# Override default output directory
export CONTENT_PIPELINE_OUTPUT_DIR="/path/to/transcripts"

# Set default log level
export CONTENT_PIPELINE_LOG_LEVEL="info"
```

## Configuration Options

### Core Settings
- `engine`: Transcription engine (`whisper-local`, `whisper-api`, `auto`)
- `output_dir`: Default output directory for transcripts
- `log_level`: Logging verbosity (`debug`, `info`, `warning`, `error`)
- `language`: Default language hint (ISO 639-1 codes)

### Engine-Specific Settings

#### Local Whisper (`whisper_local`)
- `model`: Model size (`tiny`, `base`, `small`, `medium`, `large`)
- `device`: Processing device (`cpu`, `cuda`, `auto`)
- `timeout`: Operation timeout in seconds
- `retry_attempts`: Number of retry attempts

#### OpenAI API (`whisper_api`)
- `api_key`: OpenAI API key (use environment variable)
- `model`: API model (`whisper-1`)
- `temperature`: Transcription temperature (0.0-1.0)
- `timeout`: API request timeout

### Auto-Selection Settings
- `auto_prefer_local`: Prefer local processing for privacy
- `auto_fallback_enabled`: Allow fallback to other engines
- `auto_priority_order`: Custom engine priority list

## Environment Variables

### Service Credentials
- `OPENAI_API_KEY`: OpenAI API key for whisper-api engine
- `AWS_ACCESS_KEY_ID`: AWS credentials for aws-transcribe
- `AWS_SECRET_ACCESS_KEY`: AWS credentials for aws-transcribe
- `AWS_DEFAULT_REGION`: AWS region for aws-transcribe

### Content Pipeline Variables
- `CONTENT_PIPELINE_DEFAULT_ENGINE`: Default engine selection
- `CONTENT_PIPELINE_OUTPUT_DIR`: Default output directory
- `CONTENT_PIPELINE_LOG_LEVEL`: Default logging level
- `CONTENT_PIPELINE_LOG_FILE`: Log file path (if file logging enabled)

### Engine-Specific Variables
- `WHISPER_LOCAL_MODEL`: Default local Whisper model
- `AWS_TRANSCRIBE_BUCKET`: S3 bucket for AWS Transcribe

## Variable Substitution

Configuration files support environment variable substitution:

```yaml
# Use environment variable with default fallback
output_dir: ${CONTENT_PIPELINE_OUTPUT_DIR:-./transcripts}

# Use environment variable (required)
api_key: ${OPENAI_API_KEY}

# Use system environment variables
temp_dir: ${TMPDIR:-/tmp}
log_file: ${HOME}/logs/content-pipeline.log
```

## Common Use Cases

### Privacy-Focused Setup
```yaml
engine: local-whisper
whisper_local:
  model: medium
auto_prefer_local: true
auto_fallback_enabled: false
```

### Quality-Focused Setup
```yaml
engine: openai-whisper
whisper_api:
  temperature: 0.0
auto_prefer_local: false
```

### Development Setup
```yaml
engine: auto
log_level: debug
whisper_local:
  model: tiny  # Fastest for testing
logging:
  enable_file_logging: true
```

### Production Setup
```yaml
engine: auto
log_level: warning
output_dir: /var/transcripts
logging:
  enable_file_logging: true
  log_file: /var/log/content-pipeline.log
monitoring:
  enable_metrics: true
```

## Validation and Troubleshooting

### Validate Configuration
```bash
# Test configuration loading
content-pipeline transcribe --engine auto --help

# Debug configuration resolution
content-pipeline transcribe --engine auto --log-level debug --source test.mp3
```

### Common Issues

1. **Invalid YAML Syntax**
   - Use spaces, not tabs for indentation
   - Ensure quotes are properly closed
   - Validate at https://yamlchecker.com/

2. **Missing Environment Variables**
   - Check variable names are correct
   - Use `${VAR:-default}` syntax for optional variables
   - Set required variables before running

3. **Permission Issues**
   - Ensure output directories are writable
   - Check log file directory permissions
   - Verify config file permissions

4. **Engine Not Available**
   - Install required dependencies
   - Check API credentials
   - Use `--engine auto` for automatic selection

## Security Best Practices

1. **Never hardcode API keys** in configuration files
2. **Use environment variables** for sensitive data
3. **Set appropriate file permissions** on config files (600)
4. **Use separate configs** for different environments
5. **Enable audit logging** in production
6. **Regularly rotate API keys** and credentials

## Migration from v0.6.0

If you're upgrading from v0.6.0, the main changes are:

1. **Engine selection is now required** - add `--engine` flag or set in config
2. **Output paths are configurable** - no longer hardcoded to `./output/`
3. **Configuration files are supported** - create configs to avoid repeating options

See the main documentation for detailed migration guidance.