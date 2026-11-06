"""
API security tests.

Verifies that the API-key dependency rejects missing/invalid credentials,
accepts a valid header, is a no-op when authentication is disabled, and that
the CORS middleware allows the dashboard origin.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.repositories.database import (
    DatabaseUnavailableError,
)
from src.security import require_api_key

client = TestClient(app)


def test_api_key_rejects_missing_header(monkeypatch):
    monkeypatch.setattr(
        "src.security.config.API_KEY", "secret"
    )
    response = client.get("/stores")
    assert response.status_code == 401


def test_api_key_accepts_valid_header(monkeypatch):
    monkeypatch.setattr(
        "src.security.config.API_KEY", "secret"
    )
    with patch(
        "src.repositories.inventory.db_cursor",
        side_effect=DatabaseUnavailableError("no db"),
    ):
        response = client.get(
            "/stores", headers={"X-API-Key": "secret"}
        )
    # Authentication passed; the stubbed database returns 503.
    assert response.status_code == 503


def test_cors_preflight_allows_dashboard_origin():
    response = client.options(
        "/stores",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "http://localhost:8501"
    )


def test_require_api_key_is_noop_without_configuration(
    monkeypatch,
):
    monkeypatch.setattr("src.security.config.API_KEY", "")
    assert asyncio.run(require_api_key(None)) is None
