# EnrichmentV1 Schema Reference

**Status:** Frozen as of v1.0.0 — no breaking changes allowed.

## Structure

```json
{
    "enrichment_version": "v1",
    "metadata": {
        "provider": "string (required)",
        "model": "string (required)",
        "timestamp": "datetime (required)",
        "cost_usd": "float (required, >= 0)",
        "tokens_used": "int (required, >= 0)",
        "enrichment_types": ["string (required, min 1)"],
        "cache_hit": "bool (default: false)"
    },
    "summary": { "short": "string", "medium": "string", "long": "string" },
    "tags": { "categories": ["string"], "keywords": ["string"], "entities": ["string"] },
    "chapters": [{ "title": "string", "start_time": "string", "end_time": "string", "summary": "string" }],
    "highlights": [{ "text": "string", "timestamp": "string", "context": "string" }]
}
```

## Fields

### Top-level
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| enrichment_version | string | yes | Always "v1" |
| metadata | object | yes | Enrichment operation metadata |
| summary | object | no | Summary enrichment (at least one enrichment type required) |
| tags | object | no | Tag enrichment |
| chapters | array | no | Chapter enrichment |
| highlights | array | no | Highlight enrichment |

### metadata
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| provider | string | yes | LLM provider (openai, bedrock, claude, ollama) |
| model | string | yes | Specific model used |
| timestamp | datetime | yes | UTC timestamp |
| cost_usd | float | yes | Cost in USD (0.0 for local) |
| tokens_used | int | yes | Total tokens consumed |
| enrichment_types | array | yes | Types performed (min 1) |
| cache_hit | bool | no | Whether result was cached |
