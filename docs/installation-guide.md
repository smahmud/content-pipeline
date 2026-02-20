# Installation & Dependency Guide

This guide provides comprehensive instructions for installing the Content Pipeline and all its dependencies, along with validation steps to ensure everything is working correctly.

## 📋 Overview

The Content Pipeline requires both **Python packages** and **external system tools** to function properly. This guide covers:

- Python package dependencies
- External system tools (FFmpeg)
- Installation verification
- Troubleshooting common issues

---

## 🐍 Python Dependencies

### Core Python Packages

The following packages are automatically installed when you install the Content Pipeline:

| Package | Version | Purpose |
|---------|---------|---------|
| **click** | 8.3.0 | CLI framework and command-line interface |
| **colorama** | 0.4.6 | Cross-platform colored terminal output |
| **moviepy** | 2.2.1 | Video processing and audio extraction from local files |
| **yt-dlp** | 2025.10.22 | YouTube and streaming platform audio/video downloading |
| **pydantic** | ≥2.0 | Data validation and schema management |
| **whisper** | ≥1.0 | OpenAI Whisper for speech-to-text transcription |
| **ffmpeg-python** | ≥0.2.0 | Python wrapper for FFmpeg operations |
| **openai** | ≥1.0 | OpenAI GPT models for enrichment (optional) |
| **anthropic** | ≥0.77.0 | Anthropic Claude models for enrichment (optional) |
| **boto3** | ≥1.34.0 | AWS Bedrock for enrichment (optional) |
| **tiktoken** | ≥0.5.0 | Token counting for cost estimation (optional) |
| **pyyaml** | ≥6.0 | YAML configuration and prompt templates |
| **jinja2** | ≥3.0 | Prompt template rendering |
| **requests** | ≥2.31.0 | HTTP client for Ollama (optional) |
| **mcp** | ≥1.0 | Model Context Protocol SDK for MCP server (optional) |
| **fastapi** | ≥0.100 | REST API web framework (optional) |

### Installation Methods

#### Method 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/content-pipeline.git
cd content-pipeline

# Install Python dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Method 2: Direct Package Installation

```bash
# Install directly from source
pip install git+https://github.com/your-org/content-pipeline.git
```

#### Method 3: Production Installation

```bash
# For production environments
pip install content-pipeline
```

---

## 🛠️ External System Dependencies

### FFmpeg (Required)

FFmpeg is essential for audio/video processing and must be installed separately.

#### Windows Installation

**Option 1: Using Windows Package Manager (Recommended)**
```bash
# Run as Administrator
winget install --id=Gyan.FFmpeg -e

# Restart your terminal after installation
```

**Option 2: Manual Installation**
1. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart your terminal

#### macOS Installation

```bash
# Using Homebrew (recommended)
brew install ffmpeg

# Using MacPorts
sudo port install ffmpeg
```

#### Linux Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL/Fedora:**
```bash
# CentOS/RHEL
sudo yum install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S ffmpeg
```

---

## ✅ Installation Verification

### Step 1: Verify Python Package Installation

Run this comprehensive verification script:

```bash
python -c "
import sys
print('🔍 Verifying Python Dependencies...\n')

# Check Python version
print(f'Python Version: {sys.version}')
print()

# Check each dependency
dependencies = [
    ('click', 'CLI framework'),
    ('colorama', 'Terminal colors'),
    ('moviepy', 'Video processing'),
    ('yt_dlp', 'YouTube downloading'),
    ('pydantic', 'Data validation'),
    ('whisper', 'Speech recognition'),
    ('ffmpeg', 'FFmpeg Python wrapper')
]

all_good = True
for package, description in dependencies:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {package:15} {version:15} - {description}')
    except ImportError as e:
        print(f'❌ {package:15} {'NOT FOUND':15} - {description}')
        all_good = False

print()
if all_good:
    print('🎉 All Python dependencies are installed correctly!')
else:
    print('⚠️  Some dependencies are missing. Run: pip install -r requirements.txt')
"
```

### Step 2: Verify FFmpeg Installation

```bash
# Check FFmpeg version
ffmpeg -version

# Expected output should start with:
# ffmpeg version X.X.X ...
```

### Step 3: Verify Content Pipeline CLI

```bash
# Test CLI installation
python -m cli --version
# Expected: content-pipeline, version 0.8.7

# Test CLI help
python -m cli --help
# Should show main CLI help with extract, transcribe, enrich, and format commands

# Test subcommand help
python -m cli extract --help
python -m cli transcribe --help
python -m cli enrich --help
python -m cli format --help
```

**Note:** In v0.7.0, the transcribe command requires explicit engine selection via `--engine` flag. Available engines: local-whisper, openai-whisper, aws-transcribe, auto. The enrich command supports multiple LLM providers: openai, claude, bedrock, ollama, auto. See [cli-commands.md](cli-commands.md) for usage examples.

**Test v0.7.0 engine selection and enrichment:**
```bash
# Verify --engine option is available and required
python -m cli transcribe --help
# Should show --engine option as required with choices: local-whisper, openai-whisper, aws-transcribe, auto

# Verify enrichment command is available
python -m cli enrich --help
# Should show enrichment options including --provider, --quality, --summarize, --tag, etc.
```

### Step 4: Verify Core Functionality

```bash
# Test that core imports work
python -c "
from cli.extract import extract
from cli.transcribe import transcribe
from cli.enrich import enrich
from pipeline.extractors.local.file_audio import extract_audio_from_file
from pipeline.extractors.youtube.extractor import YouTubeExtractor
print('✅ All core modules import successfully')
"
```

### Step 5: Verify Enrichment Functionality (Optional)

The enrichment system requires provider-specific API keys. Test with your preferred provider:

```bash
# Test enrichment imports
python -c "
from cli.enrich import enrich
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator
from pipeline.llm.factory import LLMProviderFactory
print('✅ Enrichment modules import successfully')
"

# Test with OpenAI (requires OPENAI_API_KEY)
# export OPENAI_API_KEY=your_key_here  # Linux/macOS
# set OPENAI_API_KEY=your_key_here     # Windows CMD
python -m cli enrich --help

# Test with Anthropic (requires ANTHROPIC_API_KEY)
# export ANTHROPIC_API_KEY=your_key_here  # Linux/macOS
# set ANTHROPIC_API_KEY=your_key_here     # Windows CMD
python -m cli enrich --help

# Test with Ollama (requires local Ollama server)
python -m cli enrich --help
```

**Note:** Enrichment providers are optional. You can use the pipeline without enrichment, or install only the providers you need:
- OpenAI: `pip install openai tiktoken`
- Anthropic: `pip install anthropic`
- AWS Bedrock: `pip install boto3`
- Ollama: No additional packages (uses local server)

---

## 🧪 Running Tests

### Test Dependencies Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Run Test Suite

```bash
# Run all tests
python -m pytest -v

# Run specific test categories
python -m pytest tests/cli/ -v                    # CLI tests
python -m pytest tests/property_tests/ -v         # Property-based tests
python -m pytest tests/integration/ -v            # Integration tests
```

### Expected Test Results

With all dependencies properly installed, you should see:

- **CLI Tests:** 28/28 passing (100%)
- **Property Tests:** 82/84 passing (97.6%)
- **Overall Success Rate:** ~98%

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. FFmpeg Not Found

**Error:** `ffmpeg: The term 'ffmpeg' is not recognized...`

**Solutions:**
- **Windows:** Restart your terminal after FFmpeg installation
- **All platforms:** Verify FFmpeg is in your system PATH
- **Manual fix:** Add FFmpeg installation directory to PATH environment variable

**Verification:**
```bash
# Should show FFmpeg version
ffmpeg -version

# If not working, check PATH
echo $PATH  # Linux/macOS
echo $env:PATH  # Windows PowerShell
```

#### 2. Python Package Import Errors

**Error:** `ModuleNotFoundError: No module named 'package_name'`

**Solutions:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check if package is installed
pip list | grep package_name

# Install specific package
pip install package_name
```

#### 3. MoviePy Version Issues

**Error:** Version mismatch or import errors with MoviePy

**Solutions:**
```bash
# Install exact version
pip install moviepy==2.2.1

# Verify installation
python -c "import pkg_resources; print(pkg_resources.get_distribution('moviepy').version)"
```

#### 4. Whisper Installation Issues

**Error:** Whisper fails to install or import

**Solutions:**
```bash
# Install with specific version
pip install openai-whisper

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Permission Issues (Windows)

**Error:** Permission denied during installation

**Solutions:**
- Run terminal as Administrator
- Use `--user` flag: `pip install --user -r requirements.txt`
- Check antivirus software isn't blocking installation

#### 6. Network/Proxy Issues

**Error:** Connection timeouts during package installation

**Solutions:**
```bash
# Use different index
pip install -r requirements.txt -i https://pypi.org/simple/

# Configure proxy (if needed)
pip install -r requirements.txt --proxy http://proxy.company.com:8080
```

---

## 🚀 Quick Validation Script

Save this as `validate_installation.py` for quick dependency checking:

```python
#!/usr/bin/env python3
"""
Content Pipeline Installation Validator

Run this script to verify all dependencies are properly installed.
"""

import sys
import subprocess
import importlib.util

def check_python_package(package_name, description):
    """Check if a Python package is installed and importable."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, "Not found"
        
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def check_system_command(command, description):
    """Check if a system command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        else:
            return False, "Command failed"
    except FileNotFoundError:
        return False, "Command not found"
    except subprocess.TimeoutExpired:
        return False, "Command timeout"
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Content Pipeline Installation Validator")
    print("=" * 50)
    
    # Python packages to check
    python_deps = [
        ('click', 'CLI framework'),
        ('colorama', 'Terminal colors'),
        ('moviepy', 'Video processing'),
        ('yt_dlp', 'YouTube downloading'),
        ('pydantic', 'Data validation'),
        ('whisper', 'Speech recognition'),
        ('ffmpeg', 'FFmpeg Python wrapper'),
        ('yaml', 'YAML configuration'),
        ('jinja2', 'Template rendering'),
    ]
    
    # System commands to check
    system_deps = [
        ('ffmpeg', 'Audio/video processing'),
        ('python', 'Python interpreter'),
    ]
    
    print("\n📦 Python Dependencies:")
    python_ok = True
    for package, desc in python_deps:
        success, version = check_python_package(package, desc)
        status = "✅" if success else "❌"
        print(f"{status} {package:15} {version:20} - {desc}")
        if not success:
            python_ok = False
    
    print("\n🛠️  System Dependencies:")
    system_ok = True
    for command, desc in system_deps:
        success, version = check_system_command(command, desc)
        status = "✅" if success else "❌"
        version_short = version[:50] + "..." if len(version) > 50 else version
        print(f"{status} {command:15} {version_short:20} - {desc}")
        if not success:
            system_ok = False
    
    print("\n🧪 CLI Functionality:")
    try:
        # Test CLI import
        from cli import main
        print("✅ CLI import successful")
        
        # Test core functionality imports
        from pipeline.extractors.local.file_audio import extract_audio_from_file
        from pipeline.extractors.youtube.extractor import YouTubeExtractor
        print("✅ Core functionality imports successful")
        
        cli_ok = True
    except Exception as e:
        print(f"❌ CLI functionality failed: {e}")
        cli_ok = False
    
    print("\n" + "=" * 50)
    if python_ok and system_ok and cli_ok:
        print("🎉 All dependencies are properly installed!")
        print("✅ Content Pipeline is ready to use.")
        return 0
    else:
        print("⚠️  Some dependencies are missing or not working.")
        print("📖 Please refer to the installation guide for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Usage:**
```bash
python validate_installation.py
```

---

## 📚 Additional Resources

### Documentation Links
- [FFmpeg Official Documentation](https://ffmpeg.org/documentation.html)
- [OpenAI Whisper Documentation](https://github.com/openai/whisper)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
- [MoviePy Documentation](https://zulko.github.io/moviepy/)

### Getting Help
- **Issues:** Report problems on [GitHub Issues](https://github.com/your-org/content-pipeline/issues)
- **Discussions:** Join [GitHub Discussions](https://github.com/your-org/content-pipeline/discussions)
- **Development:** See [CLI Commands Guide](cli-commands.md) for development information

### Version Compatibility
- **Python:** 3.8+ (recommended: 3.10+)
- **Operating Systems:** Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **Architecture:** x64 (ARM64 support varies by dependency)