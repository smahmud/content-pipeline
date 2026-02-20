"""Tests for mcp_server.config"""

import os
import pytest
import yaml

from mcp_server.config import MCPServerConfig


class TestMCPServerConfig:
    def test_defaults(self):
        config = MCPServerConfig()
        assert config.transport == "stdio"
        assert config.port == 8080
        assert config.log_level == "WARNING"
        assert "extract" in config.enabled_tools
        assert "validate" in config.enabled_tools
        assert len(config.enabled_tools) == 6

    def test_load_defaults_no_file(self):
        config = MCPServerConfig.load("/nonexistent/path.yaml")
        assert config.transport == "stdio"

    def test_load_from_yaml(self, tmp_path):
        cfg = {
            "mcp_server": {
                "transport": "sse",
                "port": 9090,
                "log_level": "DEBUG",
                "enabled_tools": ["validate", "extract"],
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        config = MCPServerConfig.load(str(cfg_path))
        assert config.transport == "sse"
        assert config.port == 9090
        assert config.log_level == "DEBUG"
        assert config.enabled_tools == ["validate", "extract"]

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        cfg = {"mcp_server": {"transport": "sse", "port": 9090}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        monkeypatch.setenv("MCP_TRANSPORT", "stdio")
        monkeypatch.setenv("MCP_PORT", "7070")

        config = MCPServerConfig.load(str(cfg_path))
        assert config.transport == "stdio"  # env overrides yaml
        assert config.port == 7070
