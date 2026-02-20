"""Tests for MCP server tool registration."""

import pytest

from mcp_server.server import mcp


class TestServerRegistration:
    def test_server_name(self):
        assert mcp.name == "content-pipeline"

    def test_tools_registered(self):
        """Verify all 6 tools are registered on the FastMCP instance."""
        # FastMCP stores tools internally; we check by listing them
        # The tool names should match our @mcp.tool() decorated functions
        tool_names = set()
        for tool in mcp._tool_manager._tools.values():
            tool_names.add(tool.name)

        expected = {"extract", "transcribe", "enrich", "format_content", "validate", "run_pipeline"}
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"

    def test_tool_count(self):
        """Should have at least 6 tools."""
        assert len(mcp._tool_manager._tools) >= 6
