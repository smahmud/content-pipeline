# 🧠 Content Pipeline

A modular pipeline for extracting, enriching, and publishing media content from platforms like YouTube, TikTok, and Vimeo. Designed for transparency, auditability, and enterprise-grade scalability.

---

## 🚀 Overview

This project orchestrates audio extraction, transcription, metadata enrichment, and publishing workflows across multiple platforms. It supports CLI invocation, schema enforcement, and future MCP server integration. Powered by AI transcription and LLM enrichment for intelligent content transformation and multi-format publishing.

For installation and setup, see [installation-guide.md](installation-guide.md).  
For CLI commands and usage, see [cli-commands.md](cli-commands.md).  
For system internals, see [architecture.md](architecture.md).  
For folder layout, see [project-structure.md](project-structure.md).  
For testing strategy and coverage, see [test-strategy.md](test-strategy.md).

---

## 📦 Key Features

- Multi-platform content extraction and ingestion (YouTube, TikTok, Vimeo, local files, cloud storage)
- AI-powered transcription with multiple provider support (local-whisper, openai-whisper, aws-transcribe, auto) with YAML configuration and environment variables
- LLM-driven content enrichment and semantic analysis (summarization, tagging, chapters, highlights)
- Multi-format content generation and publishing (blogs, tweet threads, SEO metadata, social media)
- Enterprise-grade architecture with schema validation and audit trails
- Modular CLI-first design with GUI and API interfaces
- Flexible storage and deployment options (local, cloud, hybrid, multi-provider)

---

## 📈 Milestone Status

### ✅ Completed
- Initial CLI-based YouTube audio extractor with MP3 conversion
- Local video support for MP4 ingestion
- Retry logic and logging hardening with yt-dlp integration
- Post-merge cleanup and changelog recovery
- Metadata extraction and schema enforcement
- Architecture overhaul and modular design
- Transcriber functionality with Whisper providers, transcript normalization, and schema persistence
- Modular CLI architecture with `cli/` package, shared options, extensible subcommands, and comprehensive testing
- Enhanced transcription and configuration with multiple provider support (local-whisper, openai-whisper, aws-transcribe, auto), explicit provider selection, YAML configuration management, environment variable support, and user-controlled output paths
- LLM-powered enrichment with semantic analysis for summaries, tags, chapters, and highlights; multi-provider support (OpenAI, Claude, Bedrock, Ollama); cost control and dry-run mode; quality presets and content profiles
- **v0.7.5** - Infrastructure refactoring: unified provider architecture, renamed "adapters" and "agents" to "providers", configuration management system
- **v0.7.6** - Documentation fixes: updated all documentation to reflect v0.7.5 terminology and architecture changes

### 🔧 In Progress
- **v0.8.0** - Formatter + publishing draft generator with 16 output types, hybrid templates + LLM approach, style profiles, and platform validation

### 🧭 Future
- Additional features and enhancements planned

---

## � License

This project is licensed under  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

You may:
- Share and adapt the material with attribution
- Not use it for commercial purposes
- Not use it for training machine learning models (including LLMs) without explicit permission

See [LICENSE.md](../LICENSE.md) for full legal terms.  
Full license text: [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
</content>
</invoke>