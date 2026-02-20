"""Tests for API request/response models."""

import pytest
from pydantic import ValidationError

from api.models import (
    ExtractRequest,
    FormatRequest,
    ValidateRequest,
    PipelineResponse,
    HealthResponse,
)


class TestRequestModels:
    def test_extract_request_valid(self):
        req = ExtractRequest(source="https://youtube.com/watch?v=abc")
        assert req.source == "https://youtube.com/watch?v=abc"
        assert req.output_path is None

    def test_extract_request_missing_source(self):
        with pytest.raises(ValidationError):
            ExtractRequest()

    def test_format_request_valid(self):
        req = FormatRequest(input_path="enriched.json", output_type="blog")
        assert req.output_type == "blog"
        assert req.llm_enhance is True

    def test_format_request_missing_required(self):
        with pytest.raises(ValidationError):
            FormatRequest(input_path="enriched.json")  # missing output_type

    def test_validate_request_defaults(self):
        req = ValidateRequest(input_path="file.json")
        assert req.schema_type == "auto"
        assert req.strict is False


class TestResponseModels:
    def test_pipeline_response_success(self):
        resp = PipelineResponse(success=True, tool="validate", result={"is_valid": True})
        assert resp.success is True

    def test_pipeline_response_error(self):
        resp = PipelineResponse(success=False, tool="extract", error="File not found")
        assert resp.error == "File not found"

    def test_health_response(self):
        resp = HealthResponse(version="0.10.0", tools=["extract", "validate"])
        assert resp.status == "ok"
        assert len(resp.tools) == 2
