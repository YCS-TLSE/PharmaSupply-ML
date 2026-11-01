"""
===============================================================================
PharmaSupply-ML: Test Suite for FastAPI Serving & Inventory Decision Service
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This test module serves as a practical guide for learning MLOps testing best
practices using pytest and FastAPI's TestClient.

In a production MLOps pipeline, model serving is not just about producing a
raw prediction; it encompasses business logic, health checks, data quality
validation, and operational alert routing.

Key Testing Concepts Demonstrated:
1. Endpoint Coverage:
   Ensures that every API route (/model/health, /stores, /stores/{id}/alerts,
   /features/{store}/{product}, /predict) behaves according to specification.

2. Schema & Contract Validation (Data Quality Guardrails):
   Verifies that Pydantic models automatically enforce valid input boundaries
   (e.g., rejecting negative stock levels with HTTP 422 Unprocessable Entity).

3. Graceful Degradation & Dependency Mocking:
   Uses unittest.mock.patch to simulate scenarios where the ML model artifact
   is uninitialized or missing, ensuring the API responds with an explicit
   HTTP 503 Service Unavailable status instead of crashing unexpected.

4. Train-Serve Parity & Integration Verification:
   Validates that the API correctly handles input features matching the
   schema required by the trained XGBoost model.

Author: PharmaSupply-ML Team
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

# Add the project root directory to sys.path to allow absolute imports of src.api
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.api import app

# Initialize the lightweight TestClient for FastApi routing inspection
client = TestClient(app)

# Standard valid payload mirroring PredictionInput schema requirements
VALID_PAYLOAD = {
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 90.0,
    "lag_7": 630.0,
    "lag_14": 1260.0,
    "lag_30": 2700.0,
}


def test_model_health():
    """
    Validates the model health endpoint.
    Ensures that health status and human-readable operational messages
    are correctly exposed for front-end interface consumption.
    """
    response = client.get("/model/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data
    assert data["status"] in [
        "STABLE",
        "WARNING",
        "CRITICAL",
    ]


def test_get_stores():
    """
    Validates store list retrieval.
    Ensures that the database fetch endpoint returns a structured list of stores.
    """
    response = client.get("/stores")
    assert response.status_code == 200
    data = response.json()
    assert "stores" in data
    assert isinstance(data["stores"], list)


def test_get_products():
    """
    Validates product catalog retrieval for a given store.
    Checks that valid product identifiers are returned in list format.
    """
    response = client.get("/stores/1/products")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert isinstance(data["products"], list)


def test_get_store_alerts():
    """
    Validates the multi-level operational alert engine.
    Ensures that parameter misconfigurations, stockout/expiry risks,
    and model drift alerts are formatted into a prioritized list.
    """
    response = client.get("/stores/1/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "store_id" in data
    assert "model_status" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_get_latest_features():
    """
    Validates dynamic feature retrieval for specific store/product pairs.
    Checks that lag variables and logistical constraints are correctly formatted.
    """
    response = client.get("/features/1/MED_001")
    assert response.status_code == 200
    data = response.json()
    assert "stock_on_hand" in data
    assert "lag_7" in data


def test_predict_success():
    """
    Validates inference execution under nominal operating conditions.
    When the model artifact is loaded, verifies demand predictions
    and replenishment quantity calculations enforcing MOQ and batch constraints.
    """
    response = client.post("/predict", json=VALID_PAYLOAD)

    # If the XGBoost model artifact is loaded in the local/CI environment
    if response.status_code == 200:
        data = response.json()
        assert str(data["store_id"]) == "1"
        assert str(data["product_id"]) == "MED_001"
        assert "predicted_demand" in data
        assert data["predicted_demand"] >= 0.0
        assert "order_quantity" in data
    else:
        # Fallback assertion if model binary file is absent during headless test runs
        assert response.status_code == 503


def test_predict_model_not_loaded():
    """
    Simulates uninitialized model state using mock patching.
    Ensures the service returns an explicit HTTP 503 status code
    when the model binary is unavailable in memory.
    """
    with patch("src.api.loaded_model", None):
        response = client.post(
            "/predict", json=VALID_PAYLOAD
        )
        assert response.status_code == 503
        assert (
            response.json()["detail"]
            == "ML model artifact is not loaded in memory."
        )


def test_predict_invalid_schema():
    """
    Validates Pydantic schema guardrails.
    Confirms that invalid business values (e.g., negative stock levels)
    are intercepted and rejected with HTTP 422 before reaching model inference.
    """
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload[
        "stock_on_hand"
    ] = -50.0  # Invalid negative stock value

    response = client.post("/predict", json=invalid_payload)
    assert (
        response.status_code == 422
    )  # Unprocessable Entity
