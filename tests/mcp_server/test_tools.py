"""Tests for MCP tool wrappers.

Tests the validate and extract tools directly (they don't require
external services). Other tools are tested with mocked dependencies.
"""

import json
import pytest

from mcp_server.tools.validate import validate
from mcp_server.tools.extract import extract


class TestValidateTool:
    @pytest.mark.asyncio
    async def test_validate_valid_enrichment(self, tmp_path):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-02-20T10:00:00Z",
                "cost_usd": 0.0,
                "tokens_used": 0,
                "enrichment_types": ["summary"],
            },
            "summary": {"short": "S", "medium": "M", "long": "L"},
        }
        fp = tmp_path / "enriched.json"
        fp.write_text(json.dumps(data))

        result = await validate(input_path=str(fp))
        assert result["success"] is True
        assert result["is_valid"] is True
        assert result["schema_type"] == "enrichment"

    @pytest.mark.asyncio
    async def test_validate_invalid_file(self):
        result = await validate(input_path="/nonexistent/file.json")
        assert result["success"] is True  # tool succeeded, but file is invalid
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_with_platform(self, tmp_path):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-02-20T10:00:00Z",
                "cost_usd": 0.0,
                "tokens_used": 0,
                "enrichment_types": ["summary"],
            },
            "summary": {"short": "S", "medium": "M", "long": "L"},
        }
        fp = tmp_path / "enriched.json"
        fp.write_text(json.dumps(data))

        result = await validate(input_path=str(fp), platform="twitter")
        assert result["success"] is True
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_validate_strict_mode(self, tmp_path):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-02-20T10:00:00Z",
                "cost_usd": 0.0,
                "tokens_used": 0,
                "enrichment_types": ["summary"],
            },
            "summary": {"short": "S", "medium": "M", "long": "L"},
        }
        fp = tmp_path / "enriched.json"
        fp.write_text(json.dumps(data))

        result = await validate(input_path=str(fp), strict=True)
        assert result["success"] is True


class TestExtractTool:
    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self):
        result = await extract(source="/nonexistent/video.mp4")
        assert result["success"] is False
        assert "not found" in result["error"].lower()
