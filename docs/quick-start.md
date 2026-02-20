# Quick Start Guide

Get up and running with Content Pipeline in 5 minutes.

## Installation

```bash
git clone https://github.com/smahmud/content-pipeline.git
cd content-pipeline
pip install -r requirements.txt
pip install -e .
```

## Verify Installation

```bash
content-pipeline --version
content-pipeline --help
```

## Your First Pipeline Run

### 1. Extract audio from a YouTube video

```bash
content-pipeline extract --source "https://youtube.com/watch?v=YOUR_VIDEO" --output audio.mp3
```

### 2. Transcribe the audio

```bash
content-pipeline transcribe --source output/audio.mp3 --engine local-whisper
```

### 3. Enrich the transcript

```bash
content-pipeline enrich --input output/audio.json --provider auto --types summary,tags
```

### 4. Format for publishing

```bash
content-pipeline format --input output/audio-enriched.json --type blog
content-pipeline format --input output/audio-enriched.json --type tweet --platform twitter
```

### 5. Validate the output

```bash
content-pipeline validate --input output/audio-enriched.json
```

## Using the MCP Server

```bash
python -m mcp_server
```

Add to your Kiro/Claude Desktop MCP config:

```json
{
    "mcpServers": {
        "content-pipeline": {
            "command": "python",
            "args": ["-m", "mcp_server"]
        }
    }
}
```

## Using the REST API

```bash
uvicorn api.app:app --port 8000
```

Then visit `http://localhost:8000/docs` for interactive Swagger documentation.

## Next Steps

- [CLI Commands Reference](cli-commands.md)
- [Installation Guide](installation-guide.md)
- [Architecture](architecture.md)
