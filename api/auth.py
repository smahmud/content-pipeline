"""API key authentication middleware."""

from typing import Optional

from fastapi import Header, HTTPException

from api.config import APIConfig


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[str]:
    """Verify API key from X-API-Key header.

    If no API key is configured (API_KEY env var not set), auth is disabled.
    """
    config = APIConfig.load()
    if not config.api_key:
        return None  # Auth disabled
    if not x_api_key or x_api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
