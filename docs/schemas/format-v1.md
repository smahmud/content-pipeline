# FormatV1 Schema Reference

**Status:** Frozen as of v1.0.0 — no breaking changes allowed.

## Structure

```json
{
    "format_version": "v1",
    "output_type": "string (required)",
    "platform": "string (optional)",
    "timestamp": "datetime (required)",
    "source_file": "string (required)",
    "style_profile_used": "string (optional)",
    "llm_metadata": {
        "provider": "string",
        "model": "string",
        "cost_usd": "float",
        "tokens_used": "int",
        "temperature": "float",
        "enhanced": "bool"
    },
    "validation": {
        "platform": "string (optional)",
        "character_count": "int (required)",
        "truncated": "bool",
        "warnings": ["string"]
    },
    "tone": "string (optional)",
    "length": "string (optional)"
}
```

## Fields

### Top-level
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| format_version | string | yes | Always "v1" |
| output_type | string | yes | Output type (blog, tweet, linkedin, etc.) |
| platform | string | no | Target platform |
| timestamp | datetime | yes | UTC timestamp |
| source_file | string | yes | Path to source enriched JSON |
| style_profile_used | string | no | Style profile name |
| llm_metadata | object | no | Present when LLM enhancement used |
| validation | object | yes | Platform validation results |
| tone | string | no | Tone setting used |
| length | string | no | Length setting used |

### validation
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| character_count | int | yes | Final character count |
| truncated | bool | no | Whether content was truncated |
| warnings | array | no | Validation warnings |
