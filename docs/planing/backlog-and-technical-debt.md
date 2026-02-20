# Backlog, Nice-to-Have Features & Technical Debt

This document tracks postponed features, enhancement ideas, and technical debt items for future consideration.

---

## 📋 Nice-to-Have Features (Future Enhancements)

### Enrich Command Enhancements (v0.8.6)

| Feature | Description | Priority | Target |
|---------|-------------|----------|--------|
| Separate output files | Option to output summary, tags, chapters as separate files instead of single JSON | High | v0.8.6 |
| Combined output flag | Explicit flag to merge separate enrichments into single file | High | v0.8.6 |

### Format Command Enhancements (v0.8.7)

| Feature | Description | Priority | Target |
|---------|-------------|----------|--------|
| Image prompt generator | New output type that generates detailed prompts for image generation LLMs (DALL-E, Midjourney) | High | v0.8.7 |
| Blog with images | Generate blog articles with placeholder images or AI-generated image prompts | High | v0.8.7 |
| Code sample integration | Include code samples in blog/technical content outputs | High | v0.8.7 |
| Multi-source input | Accept supporting materials (PDFs, docs) alongside enriched JSON | Medium | v0.8.7 |

---

## 🔧 Technical Debt

### High Priority

| Issue | Description | Location | Added | Status |
|-------|-------------|----------|-------|--------|
| ~~Enrichment tests use old terminology~~ | ~~Tests use `agent_factory`, `create_agent` instead of `provider_factory`, `create_provider`~~ | ~~`tests/integration/test_enrichment_*.py`, `tests/property_tests/test_enrichment_properties.py`~~ | 2026-02-18 | ✅ Fixed in v0.8.5 |
| ~~Enrichment tests use old import paths~~ | ~~Tests import from `pipeline.enrichment.agents.*` instead of `pipeline.llm.*`~~ | ~~`tests/integration/test_enrichment_providers.py`~~ | 2026-02-18 | ✅ Fixed in v0.8.5 |

### Medium Priority

| Issue | Description | Location | Added | Status |
|-------|-------------|----------|-------|--------|
| `--all` flag deprecation | Deprecated in v0.7.0 due to provider-specific reliability issues (Bedrock fails) | `cli/enrich.py` | 2026-01-29 | Open |

---

## 📝 Notes

- Items are added as they're identified during development
- Priority and target versions are estimates, subject to change
- Technical debt should be addressed before adding new features when possible
