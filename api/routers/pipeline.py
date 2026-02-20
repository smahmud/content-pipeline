"""Full pipeline endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import PipelineRequest, PipelineResponse

router = APIRouter()


@router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, _key=Depends(verify_api_key)):
    """Run full pipeline: extract → transcribe → enrich → format."""
    from mcp_server.tools.pipeline import run_pipeline as _pipeline
    result = await _pipeline(
        input_url=request.input_url,
        output_types=request.output_types,
        platform=request.platform,
    )
    return PipelineResponse(
        success=result.get("success", False),
        tool="pipeline",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
