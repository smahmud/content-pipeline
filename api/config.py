"""API configuration with env var support."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class APIConfig:
    """REST API configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    debug: bool = False

    @classmethod
    def load(cls) -> "APIConfig":
        config = cls()
        if os.environ.get("API_HOST"):
            config.host = os.environ["API_HOST"]
        if os.environ.get("API_PORT"):
            config.port = int(os.environ["API_PORT"])
        if os.environ.get("API_KEY"):
            config.api_key = os.environ["API_KEY"]
        if os.environ.get("API_DEBUG"):
            config.debug = os.environ["API_DEBUG"].lower() in ("1", "true", "yes")
        if os.environ.get("API_CORS_ORIGINS"):
            config.cors_origins = os.environ["API_CORS_ORIGINS"].split(",")
        return config
