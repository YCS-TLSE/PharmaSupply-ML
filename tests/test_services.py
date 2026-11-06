"""
Unit tests for the pure inventory business services.

Covers store-ID normalisation, the replenishment algorithm (MOQ, batch
rounding, shelf-life cap), FEFO loss estimation, alert prioritisation and the
shared feature-vector builder. These tests require no database or HTTP layer.
"""

from __future__ import annotations

import pytest

from src.config import FEATURE_COLS
from src.schemas import PredictionInput
from src.services.inventory import (
    build_alerts,
    build_prediction_features,
    compute_fefo_loss,
    compute_reorder,
    features_from_rows,
    normalize_store_id,
)


def test_normalize_store_id_variants():
    assert normalize_store_id("1") == [
        "1",
        "STORE_001",
        "STORE_1",
    ]
    assert normalize_store_id("STORE_007") == [
        "7",
        "STORE_007",
    ]


def test_compute_reorder_enforces_moq_after_shelf_life_cap():
    decision = compute_reorder(
        predicted_demand=700.0,
        stock_on_hand=0.0,
        safety_threshold=150.0,
        supplier_lead_time=2.0,
        moq=400,
        batch_size=50,
        shelf_life_days=1,
    )
    assert decision.reorder_needed is True
    assert decision.order_quantity >= 400
    assert decision.order_quantity % 50 == 0


def test_compute_reorder_no_order_when_stock_is_sufficient():
    decision = compute_reorder(
        predicted_demand=70.0,
        stock_on_hand=5000.0,
        safety_threshold=100.0,
        supplier_lead_time=2.0,
        moq=10,
        batch_size=10,
        shelf_life_days=365,
    )
    assert decision.reorder_needed is False
    assert decision.order_quantity == 0


def test_compute_fefo_loss_uses_only_expiring_batches():
    batches = [
        {"quantity": 50, "expiry_days": 3},
        {"quantity": 200, "expiry_days": 60},
    ]
    assert compute_fefo_loss(120.0, batches) == 50.0
    assert compute_fefo_loss(0.0, batches) == 0.0


def test_build_alerts_prioritizes_level_and_severity():
    rows = [
        {
            "product_id": "MED_003",
            "stock_on_hand": 10,
            "target_demand": 100,
            "safety_threshold": 50,
            "critical_expiry_days": 365,
        },
        {
            "product_id": "MED_004",
            "stock_on_hand": 10,
            "target_demand": 100,
            "safety_threshold": 200,
            "critical_expiry_days": 10,
        },
    ]
    alerts = build_alerts(rows, "WARNING")
    assert alerts[0].level == 1
    assert alerts[-1].category == "DRIFT"
    levels = [alert.level for alert in alerts]
    assert levels == sorted(levels)


def test_features_from_rows_returns_history_and_typed_contract():
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
    result = features_from_rows(rows)
    assert result.critical_expiry_days == 120
    assert len(result.sales_history) == 30
    assert result.lag_7 == 100.0


def test_features_from_rows_rejects_empty_rows():
    with pytest.raises(ValueError):
        features_from_rows([])


def test_build_prediction_features_history_path():
    payload = PredictionInput(
        store_id="1",
        product_id="MED_001",
        stock_on_hand=500.0,
        supplier_lead_time=3.0,
        safety_threshold=150.0,
        lag_7=10.0,
        lag_14=10.0,
        lag_30=10.0,
        history=[100.0] * 30,
    )
    features = build_prediction_features(payload)
    assert set(features) == set(FEATURE_COLS)
    assert features["lag_7"] == 100.0
    assert features["rolling_mean_7"] == 100.0
    assert features["stock_on_hand"] == 500.0


def test_build_prediction_features_legacy_path():
    payload = PredictionInput(
        store_id="1",
        product_id="MED_001",
        stock_on_hand=500.0,
        supplier_lead_time=3.0,
        safety_threshold=150.0,
        lag_7=70.0,
        lag_14=140.0,
        lag_30=300.0,
    )
    features = build_prediction_features(payload)
    assert set(features) == set(FEATURE_COLS)
    assert features["rolling_mean_7"] == 10.0
    assert features["rolling_mean_30"] == 10.0
