# 🧠 Content Pipeline

A modular pipeline for extracting, enriching, formatting, and publishing media content from platforms like YouTube, TikTok, and Vimeo. Built for transparency, auditability, and enterprise-grade scalability.

## Quick Start

```bash
pip install -r requirements.txt && pip install -e .
content-pipeline --help
```

See [Quick Start Guide](docs/quick-start.md) for a full walkthrough.

## Features

- **Extract** audio and metadata from YouTube, local files, and streaming platforms
- **Transcribe** with multiple engines (Whisper local, OpenAI, AWS Transcribe)
- **Enrich** transcripts with LLM-powered summaries, tags, chapters, and highlights
- **Format** into 17 output types (blog, tweet, LinkedIn, YouTube, SEO, newsletters, and more)
- **Validate** artifacts against schemas and platform limits with batch support
- **MCP Server** for AI agent orchestration (Claude, GPT, Kiro)
- **REST API** with FastAPI, Swagger docs, and API key auth

## Interfaces

| Interface | Command | Description |
|-----------|---------|-------------|
| CLI | `content-pipeline <command>` | Primary interface for all operations |
| MCP Server | `python -m mcp_server` | AI agent integration via Model Context Protocol |
| REST API | `uvicorn api.app:app` | HTTP interface with Swagger docs at `/docs` |

## Documentation

- [Quick Start](docs/quick-start.md)
- [CLI Commands](docs/cli-commands.md)
- [Installation Guide](docs/installation-guide.md)
- [Architecture](docs/architecture.md)
- [Schema Reference](docs/schemas/)
- [Release Notes](docs/releases/)
- [Contributing](CONTRIBUTING.md)
