# Release Notes

This directory contains detailed release notes for each published version of the Content Pipeline.

## Current Release

- **[v0.10.0](v0.10.0.md)** - REST API (February 20, 2026)

## Previous Releases

- **[v0.9.5](v0.9.5.md)** - MCP Server Integration (February 20, 2026)
- **[v0.9.0](v0.9.0.md)** - Validation & Quality Gates (February 20, 2026)
- **[v0.8.7](v0.8.7.md)** - Format Command Enhancements (February 20, 2026)

- **[v0.8.6](v0.8.6.md)** - Enrich Separate Output Files (February 19, 2026)
- **[v0.8.5](v0.8.5.md)** - Technical Debt Fix (February 19, 2026)
- **[v0.8.0](v0.8.0.md)** - Content Formatting (February 18, 2026)
- **[v0.7.5](v0.7.5.md)** - Infrastructure Refactoring (February 3, 2026)
- **[v0.7.0](v0.7.0.md)** - LLM-Powered Enrichment (January 30, 2026)
- **[v0.6.5](v0.6.5.md)** - Enhanced Transcription & Configuration (January 29, 2026)

For older releases (v0.6.0 and earlier), see [CHANGELOG.md](../../CHANGELOG.md) for brief release notes.

## Release Documentation Format

Each release note includes:

- **Overview**: High-level summary of the release
- **What's New**: Major features and improvements
- **Breaking Changes**: Changes requiring migration
- **Upgrade Guide**: Step-by-step migration instructions
- **Configuration Examples**: Common setup patterns
- **Known Issues**: Current limitations and workarounds
- **Deprecations**: Features scheduled for removal
- **Performance Notes**: Benchmarks and recommendations
- **Security Considerations**: Best practices and warnings

## Quick Reference

For a concise changelog of all releases, see [CHANGELOG.md](../../CHANGELOG.md) in the root directory.

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking changes, major features
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

## Release Process

1. Complete milestone implementation
2. Update CHANGELOG.md with brief entries
3. Create detailed release note in docs/releases/
4. Tag release in git: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push tag: `git push origin vX.Y.Z`
6. Create GitHub release with release note content

## Support Policy

- **Current Release**: Full support and updates
- **Previous Minor**: Security fixes only
- **Older Releases**: No active support

## Feedback

For questions about releases or upgrade issues:

- Open an issue on GitHub
- Check the [Installation Guide](../installation-guide.md)
- Review the [Architecture](../architecture.md) documentation
