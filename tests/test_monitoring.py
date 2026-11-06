"""
Unit tests for drift monitoring.

Covers SHAP-weight loading and fallback, weighted score computation, threshold
mapping, atomic status persistence, Evidently parsing helpers and the
production-drift sample-size guardrail. Evidently itself is stubbed so the
tests remain fast and deterministic.
"""

from __future__ import annotations

import json

import pytest

import src.monitoring as monitoring
from src.config import FEATURE_COLS


def test_load_feature_weights_from_json(tmp_path):
    weights_path = tmp_path / "feature_weights.json"
    weights_path.write_text(
        json.dumps({"lag_7": 0.7, "stock_on_hand": 0.3}),
        encoding="utf-8",
    )
    weights = monitoring.load_feature_weights(weights_path)
    assert weights == {"lag_7": 0.7, "stock_on_hand": 0.3}


def test_load_feature_weights_falls_back(tmp_path):
    weights = monitoring.load_feature_weights(
        tmp_path / "missing.json"
    )
    assert "lag_7" in weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_weighted_drift_score():
    drift = {
        "lag_7": {"drift_detected": True},
        "stock_on_hand": {"drift_detected": False},
    }
    score = monitoring.compute_weighted_drift_score(
        drift, {"lag_7": 0.5, "stock_on_hand": 0.5}
    )
    assert score == 0.5


def test_map_drift_status_thresholds():
    assert monitoring.map_drift_status(0.0) == "STABLE"
    assert monitoring.map_drift_status(0.20) == "WARNING"
    assert monitoring.map_drift_status(0.40) == "CRITICAL"


def test_production_drift_requires_enough_samples(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        json.dumps({"lag_7": 1.0}) + "\n", encoding="utf-8"
    )
    status = monitoring.run_production_drift_monitoring(
        prediction_log_path=log_path
    )
    assert status == "UNKNOWN"


def test_write_status_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "status.txt"
    monkeypatch.setattr(
        monitoring, "STATUS_FILE_PATH", str(target)
    )
    monitoring._write_status("WARNING")
    assert target.read_text(encoding="utf-8") == "WARNING"
    assert not target.with_name("status.txt.tmp").exists()


class _FakeEvaluation:
    def __init__(self, drift):
        self._drift = drift

    def as_dict(self):
        return {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "drift_by_columns": self._drift
                    },
                }
            ]
        }


class _FakeReport:
    def __init__(self, metrics):
        self.metrics = metrics

    def run(self, reference_data, current_data):
        return _FakeEvaluation(
            {"lag_7": {"drift_detected": True}}
        )

    def save_html(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("<html></html>")


def test_extract_drift_by_column():
    evaluation = _FakeEvaluation(
        {"lag_7": {"drift_detected": True}}
    )
    assert monitoring._extract_drift_by_column(
        evaluation
    ) == {"lag_7": {"drift_detected": True}}


def test_run_report_with_fake_evidently(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "Report",
        lambda metrics: _FakeReport(metrics),
    )
    reference = monitoring.pd.DataFrame(
        {"lag_7": [1.0, 2.0]}
    )
    drift = monitoring._run_report(reference, reference)
    assert drift["lag_7"]["drift_detected"] is True


def test_production_drift_with_enough_samples(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        monitoring,
        "Report",
        lambda metrics: _FakeReport(metrics),
    )
    monkeypatch.setattr(
        monitoring,
        "load_feature_weights",
        lambda: {"lag_7": 1.0},
    )
    monkeypatch.setattr(
        monitoring, "_write_status", lambda status: None
    )

    log_path = tmp_path / "predictions.jsonl"
    row = {column: 1.0 for column in FEATURE_COLS}
    with log_path.open("w", encoding="utf-8") as handle:
        for _ in range(30):
            handle.write(json.dumps(row) + "\n")

    reference = monitoring.pd.DataFrame(
        [{column: 1.0 for column in FEATURE_COLS}]
    )
    status = monitoring.run_production_drift_monitoring(
        prediction_log_path=log_path,
        reference_df=reference,
    )
    assert status == "CRITICAL"
