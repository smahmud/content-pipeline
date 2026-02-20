"""Pydantic request/response models for the REST API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Response Models ---

class PipelineResponse(BaseModel):
    """Standard response for all pipeline endpoints."""
    success: bool
    tool: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    tools: List[str]


# --- Request Models ---

class ExtractRequest(BaseModel):
    source: str = Field(..., description="YouTube URL or local file path")
    output_path: Optional[str] = None


class TranscribeRequest(BaseModel):
    input_path: str = Field(..., description="Path to audio file")
    engine: str = "local-whisper"
    language: Optional[str] = None
    output_path: Optional[str] = None


class EnrichRequest(BaseModel):
    input_path: str = Field(..., description="Path to transcript JSON")
    provider: str = "auto"
    summarize: bool = True
    tag: bool = True
    chapterize: bool = True
    highlight: bool = True
    output_path: Optional[str] = None


class FormatRequest(BaseModel):
    input_path: str = Field(..., description="Path to enriched JSON")
    output_type: str = Field(..., description="Output format type")
    platform: Optional[str] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    style_profile: Optional[str] = None
    llm_enhance: bool = True
    output_path: Optional[str] = None


class ValidateRequest(BaseModel):
    input_path: str = Field(..., description="Path to JSON file to validate")
    schema_type: str = "auto"
    platform: Optional[str] = None
    strict: bool = False


class PipelineRequest(BaseModel):
    input_url: str = Field(..., description="YouTube URL or local file path")
    output_types: List[str] = Field(default=["blog"])
    platform: Optional[str] = None
