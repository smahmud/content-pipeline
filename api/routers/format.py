"""Format endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import FormatRequest, PipelineResponse

router = APIRouter()


@router.post("/format", response_model=PipelineResponse)
async def format_content(request: FormatRequest, _key=Depends(verify_api_key)):
    """Format enriched content into publishing-ready output."""
    from mcp_server.tools.format import format_content as _format
    result = await _format(
        input_path=request.input_path,
        output_type=request.output_type,
        platform=request.platform,
        tone=request.tone,
        length=request.length,
        style_profile=request.style_profile,
        llm_enhance=request.llm_enhance,
        output_path=request.output_path,
    )
    return PipelineResponse(
        success=result.get("success", False),
        tool="format",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
