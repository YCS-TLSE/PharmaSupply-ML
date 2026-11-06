"""
Observability tests.

Covers the aggregated prediction JSONL (write/read round-trip), tolerance to a
missing log file, and the JSON log formatter used by the API middleware.
"""

from __future__ import annotations

import json
import logging

import src.config as config
from src.logging_config import JsonFormatter
from src.observability import (
    read_prediction_log,
    record_prediction,
)


def test_record_and_read_prediction_log(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(
        config, "PREDICTION_LOG_PATH", str(log_path)
    )
    features = {
        "month": 11.0,
        "dayofweek": 2.0,
        "is_weekend": 0.0,
        "stock_on_hand": 500.0,
        "supplier_lead_time": 3.0,
        "lag_7": 100.0,
        "lag_14": 200.0,
        "lag_30": 300.0,
        "rolling_mean_7": 100.0,
        "rolling_mean_30": 100.0,
    }
    record_prediction(features, 700.0, True, "1", "MED_001")
    records = read_prediction_log(log_path)
    assert len(records) == 1
    assert records[0]["predicted_demand"] == 700.0
    assert records[0]["lag_7"] == 100.0
    assert records[0]["reorder_needed"] is True


def test_read_prediction_log_missing_file(tmp_path):
    assert (
        read_prediction_log(tmp_path / "missing.jsonl")
        == []
    )


def test_json_formatter_outputs_valid_json():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
