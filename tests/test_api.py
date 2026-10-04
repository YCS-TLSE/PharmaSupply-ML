"""
===============================================================================
PharmaSupply-ML: Test Suite for FastAPI Serving Module
===============================================================================
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

# Ajout du dossier racine au path pour importer src.api
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.api import app

client = TestClient(app)


# Payload de test valide
VALID_PAYLOAD = [
    {
        "store_id": 101,
        "product_id": 2045,
        "month": 10,
        "dayofweek": 2,
        "is_weekend": 0,
        "stock_on_hand": 500.0,
        "supplier_lead_time": 3.0,
        "lag_7": 120.5,
        "lag_14": 115.0,
        "lag_30": 110.0,
        "rolling_mean_7": 118.2,
        "rolling_mean_30": 112.4,
    }
]


def test_health_endpoint():
    """Vérifie que l'endpoint /health répond 200 OK avec la structure attendue."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_predict_success():
    """Vérifie qu'une requête valide renvoie une prédiction positive ou nulle."""
    response = client.post("/predict", json=VALID_PAYLOAD)

    # Si le modèle est chargé sur la machine de dev/CI
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["store_id"] == 101
        assert data[0]["product_id"] == 2045
        assert "predicted_demand" in data[0]
        assert data[0]["predicted_demand"] >= 0.0
    else:
        # Si le fichier .joblib n'est pas présent dans l'env de test
        assert response.status_code == 503


def test_predict_model_not_loaded():
    """Simule l'absence du modèle et vérifie le renvoi du code HTTP 503."""
    with patch("src.api.loaded_model", None):
        response = client.post(
            "/predict", json=VALID_PAYLOAD
        )
        assert response.status_code == 503
        assert (
            response.json()["detail"]
            == "ML Model is not loaded."
        )


def test_predict_invalid_schema():
    """Vérifie que Pydantic rejette un payload invalide (ex: mois hors limites)."""
    invalid_payload = [VALID_PAYLOAD[0].copy()]
    invalid_payload[0]["month"] = 13  # Mois invalide (> 12)

    response = client.post("/predict", json=invalid_payload)
    assert (
        response.status_code == 422
    )  # Unprocessable Entity
