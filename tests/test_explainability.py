"""
Unit tests for the SHAP-to-monitoring weight bridge.

Validates weight normalisation, missing-artifact handling, SHAP computation
through a stubbed TreeExplainer, and the CSV/JSON exports consumed by the
monitoring layer.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import src.explainability as explainability
from src.explainability import (
    ModelExplainer,
    normalize_feature_weights,
)


def test_normalize_feature_weights_sums_to_one():
    weights = normalize_feature_weights(
        ["lag_7", "stock_on_hand"], [1.0, 3.0]
    )
    assert weights["lag_7"] == pytest.approx(0.25)
    assert weights["stock_on_hand"] == pytest.approx(0.75)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_normalize_feature_weights_zero_total():
    weights = normalize_feature_weights(
        ["a", "b"], [0.0, 0.0]
    )
    assert weights == {"a": 0.0, "b": 0.0}


class _FakeExplanation:
    def __init__(self, values, base_values=0.0):
        self.values = values
        self.base_values = base_values


class _FakeTreeExplainer:
    def __init__(self, booster):
        self.booster = booster

    def __call__(self, frame):
        return _FakeExplanation(
            np.ones((len(frame), frame.shape[1]))
        )


def test_model_explainer_missing_artifact(tmp_path):
    explainer = ModelExplainer.__new__(ModelExplainer)
    explainer.model_path = tmp_path / "missing.joblib"
    with pytest.raises(FileNotFoundError):
        explainer._load_model()


def test_fit_explainer_and_export(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"dummy": True}, model_path)
    monkeypatch.setattr(
        explainability.shap,
        "TreeExplainer",
        _FakeTreeExplainer,
    )

    explainer = ModelExplainer(
        model_path=str(model_path),
        output_dir=str(tmp_path / "figures"),
    )
    frame = pd.DataFrame(
        {
            "lag_7": [1.0, 2.0],
            "stock_on_hand": [3.0, 4.0],
        }
    )
    explainer.fit_explainer(frame, threshold=300)
    explainer.export_shap_to_csv()
    explainer.export_feature_weights()

    weights_path = tmp_path / "feature_weights.json"
    assert weights_path.exists()
    assert (
        tmp_path / "figures" / "shap_values.csv"
    ).exists()


def test_fit_explainer_subsamples_large_input(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"dummy": True}, model_path)
    monkeypatch.setattr(
        explainability.shap,
        "TreeExplainer",
        _FakeTreeExplainer,
    )
    explainer = ModelExplainer(
        model_path=str(model_path),
        output_dir=str(tmp_path / "figures"),
    )
    frame = pd.DataFrame(
        {
            "lag_7": np.arange(400, dtype=float),
            "stock_on_hand": np.arange(400, dtype=float),
        }
    )
    explainer.fit_explainer(
        frame, threshold=300, sample_size=50, n_repeats=1
    )
    assert len(explainer.X_sample) == 50
