"""Validate endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import ValidateRequest, PipelineResponse

router = APIRouter()


@router.post("/validate", response_model=PipelineResponse)
async def validate(request: ValidateRequest, _key=Depends(verify_api_key)):
    """Validate pipeline artifacts against schemas and platform limits."""
    from mcp_server.tools.validate import validate as _validate
    result = await _validate(
        input_path=request.input_path,
        schema=request.schema_type,
        platform=request.platform,
        strict=request.strict,
    )
    return PipelineResponse(
        success=result.get("success", False),
        tool="validate",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
