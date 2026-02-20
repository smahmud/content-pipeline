"""
MCP Server Configuration

Loads server settings from YAML config or environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server."""
    transport: str = "stdio"
    port: int = 8080
    log_level: str = "WARNING"
    enabled_tools: List[str] = field(default_factory=lambda: [
        "extract", "transcribe", "enrich", "format_content", "validate", "run_pipeline",
    ])
    pipeline_config_path: str = ".content-pipeline/config.yaml"

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "MCPServerConfig":
        """Load config from YAML file, env vars, or defaults.

        Priority: env vars > YAML > defaults.
        """
        config = cls()

        # Load from YAML if available
        path = config_path or os.environ.get("MCP_SERVER_CONFIG")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            mcp_section = data.get("mcp_server", data)
            if "transport" in mcp_section:
                config.transport = mcp_section["transport"]
            if "port" in mcp_section:
                config.port = int(mcp_section["port"])
            if "log_level" in mcp_section:
                config.log_level = mcp_section["log_level"]
            if "enabled_tools" in mcp_section:
                config.enabled_tools = mcp_section["enabled_tools"]
            if "pipeline_config_path" in mcp_section:
                config.pipeline_config_path = mcp_section["pipeline_config_path"]

        # Env var overrides
        if os.environ.get("MCP_TRANSPORT"):
            config.transport = os.environ["MCP_TRANSPORT"]
        if os.environ.get("MCP_PORT"):
            config.port = int(os.environ["MCP_PORT"])
        if os.environ.get("MCP_LOG_LEVEL"):
            config.log_level = os.environ["MCP_LOG_LEVEL"]
        if os.environ.get("CONTENT_PIPELINE_CONFIG"):
            config.pipeline_config_path = os.environ["CONTENT_PIPELINE_CONFIG"]

        return config
