# Backlog, Nice-to-Have Features & Technical Debt

This document tracks postponed features, enhancement ideas, and technical debt items for future consideration.

---

## 📋 Nice-to-Have Features (Future Enhancements)

### Format Command Enhancements

| Feature | Description | Priority | Target |
|---------|-------------|----------|--------|
| Blog with images | Generate blog articles with placeholder images or AI-generated image prompts | Medium | v0.9.0+ |
| Code sample integration | Include code samples in blog/technical content outputs | Medium | v0.9.0+ |
| Multi-source input | Accept supporting materials (PDFs, docs) alongside enriched JSON | Low | v1.0+ |
| Image prompt generator | New output type that generates detailed prompts for image generation LLMs (DALL-E, Midjourney) | Medium | v0.9.0+ |

### Enrich Command Enhancements

| Feature | Description | Priority | Target |
|---------|-------------|----------|--------|
| Separate output files | Option to output summary, tags, chapters as separate files instead of single JSON | Low | v0.9.0+ |
| Combined output flag | Explicit flag to merge separate enrichments into single file | Low | v0.9.0+ |

---

## 🔧 Technical Debt

### High Priority

| Issue | Description | Location | Added |
|-------|-------------|----------|-------|
| Enrichment tests use old terminology | Tests use `agent_factory`, `create_agent` instead of `provider_factory`, `create_provider` | `tests/integration/test_enrichment_*.py`, `tests/property_tests/test_enrichment_properties.py` | 2026-02-18 |
| Enrichment tests use old import paths | Tests import from `pipeline.enrichment.agents.*` instead of `pipeline.llm.*` | `tests/integration/test_enrichment_providers.py` | 2026-02-18 |

### Medium Priority

| Issue | Description | Location | Added |
|-------|-------------|----------|-------|
| `--all` flag deprecation | Deprecated in v0.7.0 due to provider-specific reliability issues (Bedrock fails) | `cli/enrich.py` | 2026-01-29 |

---

## 📝 Notes

- Items are added as they're identified during development
- Priority and target versions are estimates, subject to change
- Technical debt should be addressed before adding new features when possible
