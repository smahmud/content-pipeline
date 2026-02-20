"""Tests for REST API endpoints."""

import json
import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Content Pipeline API"
        assert "version" in data

    def test_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "tools" in data
        assert len(data["tools"]) >= 6

    def test_version(self, client):
        resp = client.get("/api/v1/version")
        assert resp.status_code == 200
        assert "version" in resp.json()

    def test_list_tools(self, client):
        resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert "extract" in tools
        assert "validate" in tools


class TestValidateEndpoint:
    def test_validate_valid_file(self, client, tmp_path):
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-02-20T10:00:00Z",
                "cost_usd": 0.0,
                "tokens_used": 0,
                "enrichment_types": ["summary"],
            },
            "summary": {"short": "S", "medium": "M", "long": "L"},
        }
        fp = tmp_path / "enriched.json"
        fp.write_text(json.dumps(data))

        resp = client.post("/api/v1/validate", json={
            "input_path": str(fp),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["tool"] == "validate"

    def test_validate_nonexistent_file(self, client):
        resp = client.post("/api/v1/validate", json={
            "input_path": "/nonexistent/file.json",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True  # tool ran successfully
        assert body["result"]["is_valid"] is False

    def test_validate_missing_input(self, client):
        resp = client.post("/api/v1/validate", json={})
        assert resp.status_code == 422  # Pydantic validation error


class TestExtractEndpoint:
    def test_extract_nonexistent_file(self, client):
        resp = client.post("/api/v1/extract", json={
            "source": "/nonexistent/video.mp4",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["tool"] == "extract"

    def test_extract_missing_source(self, client):
        resp = client.post("/api/v1/extract", json={})
        assert resp.status_code == 422


class TestFormatEndpoint:
    def test_format_missing_fields(self, client):
        resp = client.post("/api/v1/format", json={})
        assert resp.status_code == 422  # Missing required fields
