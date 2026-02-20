"""Extract endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import ExtractRequest, PipelineResponse

router = APIRouter()


@router.post("/extract", response_model=PipelineResponse)
async def extract(request: ExtractRequest, _key=Depends(verify_api_key)):
    """Extract audio and metadata from a video URL or local file."""
    from mcp_server.tools.extract import extract as _extract
    result = await _extract(source=request.source, output_path=request.output_path)
    return PipelineResponse(
        success=result.get("success", False),
        tool="extract",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
