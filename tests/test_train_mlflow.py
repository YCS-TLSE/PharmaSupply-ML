"""
MLflow integration guardrails.

MLflow is optional and disabled by default. This test verifies that the
tracking helper is a safe no-op when disabled, so training never depends on an
external tracking server.
"""

from __future__ import annotations

import pandas as pd

import src.train as train_module


def test_mlflow_logging_is_skipped_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        train_module, "MLFLOW_ENABLED", False
    )
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert (
        train_module._log_to_mlflow(
            object(), {"mae": 1.0}, {"n": 1}, frame
        )
        is None
    )
