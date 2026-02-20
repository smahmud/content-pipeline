"""Tests for API key authentication."""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAuth:
    def test_no_auth_required_by_default(self, client):
        """When API_KEY env is not set, auth is disabled."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_auth_required_when_key_set(self, client, monkeypatch):
        """When API_KEY is set, requests without key should fail."""
        monkeypatch.setenv("API_KEY", "test-secret-key")
        resp = client.post("/api/v1/validate", json={
            "input_path": "/some/file.json",
        })
        assert resp.status_code == 401

    def test_auth_passes_with_correct_key(self, client, monkeypatch):
        """Correct API key should pass auth."""
        monkeypatch.setenv("API_KEY", "test-secret-key")
        resp = client.post(
            "/api/v1/validate",
            json={"input_path": "/some/file.json"},
            headers={"X-API-Key": "test-secret-key"},
        )
        # Should not be 401 (may be 200 with validation result)
        assert resp.status_code != 401

    def test_auth_fails_with_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-secret-key")
        resp = client.post(
            "/api/v1/validate",
            json={"input_path": "/some/file.json"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
