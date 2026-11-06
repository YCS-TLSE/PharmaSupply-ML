"""
===============================================================================
PharmaSupply-ML: Test Suite for the FastAPI Serving Layer
===============================================================================

Covers routing, explicit error handling (404/503), the typed /features
contract, the MOQ floor, liveness/readiness probes and the metrics endpoint.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import patch

import psycopg2
from fastapi.testclient import TestClient

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.api import app
from src.repositories.database import (
    DatabaseUnavailableError,
)

client = TestClient(app)

VALID_PAYLOAD = {
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 100.0,
    "lag_7": 100.0,
    "lag_14": 100.0,
    "lag_30": 100.0,
    "history": [100.0] * 30,
}


class _FakeCursor:
    """Minimal psycopg2 cursor stub returning predefined rows."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchall(self):
        return self._rows


@contextmanager
def _fake_db_cursor(rows):
    yield _FakeCursor(rows)


def _patch_db(rows):
    return patch(
        "src.repositories.inventory.db_cursor",
        side_effect=lambda dict_rows=False: _fake_db_cursor(
            rows
        ),
    )


class _FakeModel:
    """Deterministic model stub returning a 700-unit 7-day demand."""

    def predict(self, X):
        return [700.0]


def test_model_health():
    """The health endpoint is side-effect free and always typed."""
    response = client.get("/model/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in [
        "STABLE",
        "WARNING",
        "CRITICAL",
        "UNKNOWN",
    ]
    assert "message" in data


def test_livez():
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readyz_not_ready_without_model():
    with patch(
        "src.api_routers.health.check_database_health",
        return_value=True,
    ):
        response = client.get("/readyz")
    assert response.status_code == 503


def test_readyz_ready():
    app.state.model = _FakeModel()
    try:
        with patch(
            "src.api_routers.health.check_database_health",
            return_value=True,
        ):
            response = client.get("/readyz")
    finally:
        app.state.model = None

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "pharma_api_requests_total" in response.text


def test_get_stores():
    """Validates store list retrieval against a stubbed database."""
    with _patch_db([("1",), ("2",)]):
        response = client.get("/stores")

    assert response.status_code == 200
    assert response.json()["stores"] == ["1", "2"]


def test_get_stores_database_error_returns_503():
    """A database failure must be explicit, never a fake store list."""
    with patch(
        "src.repositories.inventory.db_cursor",
        side_effect=DatabaseUnavailableError("boom"),
    ):
        response = client.get("/stores")

    assert response.status_code == 503
    assert (
        "unavailable" in response.json()["detail"].lower()
    )


def test_get_products():
    """Validates product catalogue retrieval against a stubbed database."""
    with _patch_db([("MED_001",), ("MED_002",)]):
        response = client.get("/stores/1/products")

    assert response.status_code == 200
    assert response.json()["products"] == [
        "MED_001",
        "MED_002",
    ]


def test_get_store_alerts():
    """Validates the alert engine against a stubbed healthy inventory row."""
    rows = [
        {
            "product_id": "MED_001",
            "stock_on_hand": 500,
            "target_demand": 100,
            "safety_threshold": 200,
            "critical_expiry_days": 365,
            "shelf_life_days": 365,
        }
    ]
    with _patch_db(rows):
        response = client.get("/stores/1/alerts")

    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == "1"
    assert isinstance(data["alerts"], list)


def test_get_latest_features():
    """Validates the typed /features contract, including FEFO and history."""
    rows = [
        {
            "stock_on_hand": 500.0,
            "supplier_lead_time": 3.0,
            "sales_volume": 100.0,
            "safety_threshold": 150.0,
            "batch_size": 50,
            "moq": 100,
            "shelf_life_days": 365,
            "critical_expiry_days": 120,
        }
        for _ in range(30)
    ]
    with _patch_db(rows):
        response = client.get("/features/1/MED_001")

    assert response.status_code == 200
    data = response.json()
    assert data["stock_on_hand"] == 500.0
    assert data["lag_7"] == 100.0
    assert data["critical_expiry_days"] == 120
    assert len(data["sales_history"]) == 30


def test_get_latest_features_missing_history_returns_404():
    """An unknown store/product pair must return 404, not fake defaults."""
    with _patch_db([]):
        response = client.get("/features/1/UNKNOWN")

    assert response.status_code == 404


def test_predict_success(override_model):
    """Validates inference and replenishment against a deterministic model."""
    override_model(_FakeModel())
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert str(data["store_id"]) == "1"
    assert str(data["product_id"]) == "MED_001"
    assert data["predicted_demand"] == 700.0
    assert data["reorder_needed"] is True
    assert data["order_quantity"] >= VALID_PAYLOAD["moq"]
    assert (
        data["order_quantity"] % VALID_PAYLOAD["batch_size"]
        == 0
    )


def test_predict_order_quantity_respects_moq_after_shelf_life_cap(
    override_model,
):
    """The shelf-life cap must never drop the order below the supplier MOQ."""
    override_model(_FakeModel())
    payload = VALID_PAYLOAD.copy()
    payload.update(
        {
            "stock_on_hand": 0.0,
            "shelf_life_days": 1,
            "batch_size": 50,
            "moq": 400,
        }
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["order_quantity"] >= 400
    assert data["order_quantity"] % 50 == 0


def test_predict_model_not_loaded(override_model):
    """The service returns 503 when the model is unavailable."""
    override_model(None)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "ML model artifact is not loaded in memory."
    )


def test_predict_invalid_schema():
    """Pydantic rejects invalid business values with HTTP 422."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["stock_on_hand"] = -50.0
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_history_too_short_is_rejected(
    override_model,
):
    """The optional history must contain at least 30 values."""
    override_model(_FakeModel())
    payload = VALID_PAYLOAD.copy()
    payload["history"] = [100.0] * 5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_database_error_is_not_masked():
    """Sanity check: psycopg2 errors are converted to 503, not 200."""
    with patch(
        "src.repositories.database.get_db_connection",
        side_effect=psycopg2.OperationalError("down"),
    ):
        response = client.get("/stores")
    assert response.status_code == 503


def test_lifespan_loads_model_and_stops_watcher():
    """The application lifespan loads the artifact and cleans up on exit."""
    with TestClient(app) as lifespan_client:
        response = lifespan_client.get("/livez")
        assert response.status_code == 200
    assert app.state.model is None
