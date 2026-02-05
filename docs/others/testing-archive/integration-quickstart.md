# Quick Start: Full Pipeline Integration Test

## TL;DR - Run the Test

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY='your-api-key-here'

# 2. Run the test (costs ~$0.01-0.05)
pytest tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_youtube_to_enrichment -v -s --external --slow
```

## What This Test Does

1. **Extracts** audio from a short YouTube video (Python in 100 Seconds)
2. **Transcribes** the audio using Whisper (local or API)
3. **Enriches** the transcript using Claude API (generates summary)
4. **Validates** all outputs conform to schemas

**Duration**: 3-5 minutes  
**Cost**: ~$0.01-0.05 USD (uses Claude Haiku, cheapest model)

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Install ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows - download from https://ffmpeg.org/download.html
```

### 3. Set API Key
```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'
```

Get your API key from: https://console.anthropic.com/

### 4. (Optional) Install Whisper
```bash
# For local transcription (free)
pip install openai-whisper

# OR set OpenAI API key for API transcription
export OPENAI_API_KEY='sk-...'
```

## Running Tests

### Cost Estimation (FREE)
```bash
# Estimate costs without making API calls
pytest tests/integration/test_full_pipeline.py::test_pipeline_cost_estimation_only -v -s --external
```

### Full Pipeline Test (~$0.01-0.05)
```bash
# Complete workflow: extract → transcribe → enrich
pytest tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_youtube_to_enrichment -v -s --external --slow
```

### All Integration Tests
```bash
# Run all integration tests (some are free, some cost money)
pytest tests/integration/ -v -s --external --slow
```

## Expected Output

```
STAGE 1: Extracting audio from YouTube...
  ✓ Audio extracted: test_audio.mp3 (1,234,567 bytes)
  ✓ Metadata extracted: Python in 100 Seconds
  ✓ Duration: 100 seconds

STAGE 2: Transcribing audio to text...
  ✓ Using engine: local-whisper
  ✓ Transcript generated: 150 words
  ✓ Preview: Python is a high-level programming language...

STAGE 3: Enriching transcript with Claude API...
  ✓ Estimated cost: $0.0023 USD
  ✓ Generating enrichment... (this will cost money)
  ✓ Enrichment generated
  ✓ Actual cost: $0.0025 USD
  ✓ Tokens used: 1,234

STAGE 4: Validating all outputs...
  ✓ All files created successfully
  ✓ All files have valid sizes

TEST COMPLETED SUCCESSFULLY!
Video: Python in 100 Seconds
Duration: 100 seconds
Transcript: 150 words
Enrichment cost: $0.0025 USD
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### "ffmpeg not found"
Install ffmpeg (see Prerequisites above)

### "Neither local-whisper nor OPENAI_API_KEY available"
```bash
# Option 1: Install Whisper locally (free)
pip install openai-whisper

# Option 2: Use OpenAI API
export OPENAI_API_KEY='sk-...'
```

### Test Fails
```bash
# Run with more verbose output
pytest tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_youtube_to_enrichment -v -s --external --slow --tb=short

# Run with debugging
pytest tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_youtube_to_enrichment -v -s --external --slow --pdb
```

## Cost Control

The test has built-in cost controls:
- Uses **Claude Haiku** (cheapest model: $0.25/$1.25 per 1M tokens)
- Generates **only summary** (minimal enrichment)
- Has **$0.10 safety limit** (test will fail if cost exceeds this)
- Uses **short video** (~100 seconds, ~150 words)

Typical cost breakdown:
- Input tokens: ~650 (transcript)
- Output tokens: ~200 (summary)
- Total cost: **~$0.001-0.005 USD**

## What Gets Created

The test creates temporary files (automatically cleaned up):
```
/tmp/pipeline_test_XXXXXX/
├── test_audio.mp3          # Extracted audio from YouTube
├── test_audio.json         # Video metadata (title, duration, etc.)
├── test_transcript.json    # Whisper transcription result
└── test_enrichment.json    # Claude enrichment (summary)
```

## Next Steps

After running the test successfully:

1. **Explore the code**: Check `tests/integration/test_full_pipeline.py`
2. **Try different videos**: Modify `TEST_VIDEO_URL` in the test
3. **Add more enrichments**: Change `enrichment_types=["summary"]` to include tags, chapters, highlights
4. **Run CLI commands**: Use the actual CLI to process your own videos

## CLI Usage

After verifying the test works, use the CLI:

```bash
# Extract audio
content-pipeline extract --source "https://youtube.com/watch?v=..." --output video.mp3

# Transcribe
content-pipeline transcribe --input output/video.mp3 --engine local-whisper

# Enrich
content-pipeline enrich --input output/video.json --all --provider claude
```

## More Information

- [Integration Tests Documentation](integration-tests.md) - Detailed documentation
- [Testing Guide](../testing-guide.md) - Complete testing guide
- [CLI Commands](../cli-commands.md) - CLI usage documentation
