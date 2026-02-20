"""Enrich endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import EnrichRequest, PipelineResponse

router = APIRouter()


@router.post("/enrich", response_model=PipelineResponse)
async def enrich(request: EnrichRequest, _key=Depends(verify_api_key)):
    """Generate semantic enrichment from transcript."""
    from mcp_server.tools.enrich import enrich as _enrich
    result = await _enrich(
        input_path=request.input_path,
        provider=request.provider,
        summarize=request.summarize,
        tag=request.tag,
        chapterize=request.chapterize,
        highlight=request.highlight,
        output_path=request.output_path,
    )
    return PipelineResponse(
        success=result.get("success", False),
        tool="enrich",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
