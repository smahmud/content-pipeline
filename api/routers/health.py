"""Health and info endpoints."""

from fastapi import APIRouter

from api.models import HealthResponse

router = APIRouter()

TOOLS = ["extract", "transcribe", "enrich", "format", "validate", "pipeline"]
VERSION = "0.10.0"


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=VERSION, tools=TOOLS)


@router.get("/version")
async def version():
    """Return API version."""
    return {"version": VERSION}


@router.get("/tools")
async def list_tools():
    """List available pipeline tools."""
    return {"tools": TOOLS}
