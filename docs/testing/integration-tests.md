# Integration Tests

This directory contains integration tests that verify the complete content pipeline workflow with real external services.

## Overview

Integration tests cover:
- **Extract**: Audio extraction from YouTube videos
- **Transcribe**: Speech-to-text transcription using Whisper
- **Enrich**: LLM-powered semantic analysis using Claude API
- **Full Pipeline**: End-to-end workflow from video to enriched transcript

## Test Categories

### 1. Unit Tests (Mocked)
- Location: `tests/pipeline/`, `tests/cli/`
- Cost: **FREE** (no API calls)
- Speed: Fast (seconds)
- Run with: `pytest tests/`

### 2. Integration Tests (Real APIs)
- Location: `tests/integration/`
- Cost: **VARIES** (uses real APIs)
- Speed: Slow (minutes)
- Run with: `pytest tests/integration/ --external --slow`

## Running Integration Tests

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Install system dependencies**:
   - **ffmpeg**: Required for audio extraction
     ```bash
     # macOS
     brew install ffmpeg
     
     # Ubuntu/Debian
     sudo apt-get install ffmpeg
     
     # Windows
     # Download from https://ffmpeg.org/download.html
     ```
   
   - **Whisper** (optional, for local transcription):
     ```bash
     pip install openai-whisper
     ```

3. **Set API keys**:
   ```bash
   # Required for enrichment tests
   export ANTHROPIC_API_KEY='your-anthropic-api-key'
   
   # Optional: for OpenAI Whisper API transcription
   export OPENAI_API_KEY='your-openai-api-key'
   ```

### Running Tests

#### Run All Integration Tests
```bash
pytest tests/integration/ -v -s --external --slow
```

#### Run Specific Test Files
```bash
# Extract pipeline only (YouTube download)
pytest tests/integration/test_extract_pipeline_flow.py -v -s --external

# Enrichment providers only
pytest tests/integration/test_enrichment_providers.py -v -s

# Full pipeline (extract → transcribe → enrich)
pytest tests/integration/test_full_pipeline.py -v -s --external --slow
```

#### Run Cost Estimation Only (FREE)
```bash
# Dry-run mode - estimates costs without API calls
pytest tests/integration/test_full_pipeline.py::test_pipeline_cost_estimation_only -v -s --external
```

#### Run Single Full Pipeline Test
```bash
# Complete workflow with real APIs (costs ~$0.01-0.05)
pytest tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_youtube_to_enrichment -v -s --external --slow
```

### Test Markers

Tests are marked with pytest markers to control execution:

- `@pytest.mark.integration` - Integration test (may use external services)
- `@pytest.mark.external` - Requires external services (YouTube, APIs)
- `@pytest.mark.slow` - Long-running test (minutes)

Skip external tests:
```bash
pytest tests/integration/ -m "not external"
```

Skip slow tests:
```bash
pytest tests/integration/ -m "not slow"
```

## Cost Information

### Free Tests
- `test_extract_pipeline_flow.py` - YouTube extraction (free)
- `test_enrichment_providers.py` - Uses mocks (free)
- `test_pipeline_cost_estimation_only` - Dry-run mode (free)

### Paid Tests
- `test_full_pipeline_youtube_to_enrichment` - **~$0.01-0.05 USD**
  - Uses Claude Haiku (cheapest model)
  - Generates only summary (minimal enrichment)
  - Has $0.10 safety limit
  
- `test_full_pipeline_with_all_enrichments` - **~$0.05-0.15 USD**
  - Generates all enrichment types
  - Currently skipped by default
  - Run manually with `--run-expensive` flag

### Cost Breakdown (Claude Haiku)
- Input: $0.25 per 1M tokens
- Output: $1.25 per 1M tokens
- Typical 2-minute video transcript: ~500 words = ~650 tokens
- Summary generation: ~200 output tokens
- **Total cost: ~$0.001-0.005 per enrichment**

## Test Video

The full pipeline test uses a short educational video:
- **Video**: "Python in 100 Seconds" by Fireship
- **URL**: https://www.youtube.com/watch?v=kqtD5dpn9C8
- **Duration**: ~100 seconds
- **Content**: Educational, clear audio, public domain
- **Why**: Short, educational, reliable, good for testing

## Test Workflow

### Full Pipeline Test Flow

```
1. EXTRACT (30-60 seconds)
   ├─ Download audio from YouTube → test_audio.mp3
   └─ Extract metadata → test_audio.json

2. TRANSCRIBE (1-2 minutes)
   ├─ Transcribe audio with Whisper → test_transcript.json
   └─ Normalize to TranscriptV1 schema

3. ENRICH (10-30 seconds)
   ├─ Estimate cost (dry-run)
   ├─ Generate enrichment with Claude API
   └─ Save enrichment → test_enrichment.json

4. VALIDATE
   ├─ Check all files exist
   ├─ Validate schemas
   └─ Verify content quality
```

### Output Files

All test files are created in a temporary directory:
```
/tmp/pipeline_test_XXXXXX/
├── test_audio.mp3          # Extracted audio
├── test_audio.json         # Video metadata
├── test_transcript.json    # Transcription result
└── test_enrichment.json    # LLM enrichment
```

Files are automatically cleaned up after tests complete.

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### "ffmpeg not found"
Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### "Neither local-whisper nor OPENAI_API_KEY available"
Install Whisper locally:
```bash
pip install openai-whisper
```

Or set OpenAI API key:
```bash
export OPENAI_API_KEY='sk-...'
```

### "YouTube extraction failed"
- Check internet connection
- Verify video URL is accessible
- Try a different video URL

### "Cost limit exceeded"
Increase the `max_cost` parameter in the test:
```python
enrichment_request = EnrichmentRequest(
    ...
    max_cost=0.50,  # Increase from 0.10 to 0.50
)
```

## Development

### Adding New Integration Tests

1. Mark test with appropriate markers:
   ```python
   @pytest.mark.integration
   @pytest.mark.external
   @pytest.mark.slow
   def test_my_integration():
       ...
   ```

2. Use temporary workspace:
   ```python
   def test_my_integration(temp_workspace):
       # temp_workspace is automatically cleaned up
       output_file = temp_workspace / "output.json"
   ```

3. Check for API keys:
   ```python
   def test_my_integration(check_api_keys):
       # Automatically skips if API key not set
       api_key = check_api_keys
   ```

4. Document costs in docstring:
   ```python
   def test_expensive_operation():
       """
       Test expensive operation.
       
       Expected cost: ~$0.10 USD
       Expected duration: 5 minutes
       """
   ```

### Running Tests During Development

```bash
# Run only fast integration tests
pytest tests/integration/ -m "integration and not slow" -v

# Run with verbose output
pytest tests/integration/ -v -s --external --slow

# Run specific test with debugging
pytest tests/integration/test_full_pipeline.py::test_pipeline_cost_estimation_only -v -s --pdb
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Install ffmpeg
        run: sudo apt-get install -y ffmpeg
      
      - name: Install Python dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run free integration tests
        run: pytest tests/integration/ -m "integration and not external" -v
      
      - name: Run external tests (with API keys)
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration/ -m "external" -v --maxfail=1
```

## Best Practices

1. **Always run cost estimation first** before expensive tests
2. **Use cheapest models** for testing (Claude Haiku, GPT-3.5)
3. **Set cost limits** to prevent runaway expenses
4. **Clean up resources** after tests (use fixtures)
5. **Document costs** in test docstrings
6. **Skip expensive tests** by default (use markers)
7. **Use caching** to avoid redundant API calls
8. **Monitor API usage** and costs regularly

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Whisper Documentation](https://github.com/openai/whisper)
- [pytest Documentation](https://docs.pytest.org/)
- [Integration Test Quick Start](integration-quickstart.md)
- [Testing Guide](../testing-guide.md)
- [Project Documentation](../README.md)
