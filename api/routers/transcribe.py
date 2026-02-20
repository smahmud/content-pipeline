"""Transcribe endpoint."""

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import TranscribeRequest, PipelineResponse

router = APIRouter()


@router.post("/transcribe", response_model=PipelineResponse)
async def transcribe(request: TranscribeRequest, _key=Depends(verify_api_key)):
    """Transcribe audio to structured transcript."""
    from mcp_server.tools.transcribe import transcribe as _transcribe
    result = await _transcribe(
        input_path=request.input_path,
        engine=request.engine,
        language=request.language,
        output_path=request.output_path,
    )
    return PipelineResponse(
        success=result.get("success", False),
        tool="transcribe",
        result=result if result.get("success") else None,
        error=result.get("error"),
    )
