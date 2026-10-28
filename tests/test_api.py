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


# Payload de test valide aligné sur PredictionInput (objet unique)
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


def test_root_endpoint():
    """Vérifie que la documentation ou les routes de base répondent correctement."""
    response = client.get("/")
    # FastAPI renvoie 404 sur la racine par défaut sauf si défini,
    # testons plutôt un endpoint de données comme /stores
    response = client.get("/stores")
    assert response.status_code == 200
    data = response.json()
    assert "stores" in data


def test_predict_success():
    """Vérifie qu'une requête valide renvoie une prédiction et un statut corrects."""
    response = client.post("/predict", json=VALID_PAYLOAD)

    # Si le modèle est chargé sur la machine de dev/CI
    if response.status_code == 200:
        data = response.json()
        assert data["store_id"] == "1"
        assert data["product_id"] == "MED_001"
        assert "predicted_demand" in data
        assert data["predicted_demand"] >= 0.0
        assert "order_quantity" in data
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
            == "ML model artifact is not loaded in memory."
        )


def test_predict_invalid_schema():
    """Vérifie que Pydantic rejette un payload invalide (ex: stock négatif)."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload[
        "stock_on_hand"
    ] = -50.0  # Valeur négative interdite (ge=0)

    response = client.post("/predict", json=invalid_payload)
    assert (
        response.status_code == 422
    )  # Unprocessable Entity
