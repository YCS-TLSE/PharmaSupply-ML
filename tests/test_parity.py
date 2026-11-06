"""
Train/serve parity tests.

These tests prove that ``build_history_features`` (used by the API when a
``history`` payload is provided) reproduces the exact lag, rolling-mean and
calendar values computed by ``create_features`` during training. A mismatch
would silently degrade production predictions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    build_history_features,
    create_features,
)


def _synthetic_frame(days: int = 120) -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-01", periods=days, freq="D"
    )
    sales = [
        100.0 + (index % 7) * 3.0 + index * 0.5
        for index in range(days)
    ]
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": "1",
            "product_id": "MED_001",
            "sales_volume": sales,
            "is_stockout": 0,
        }
    )


def test_history_builder_matches_training_features():
    """The inference builder must reproduce the training formulas exactly."""
    frame = _synthetic_frame()
    processed = create_features(frame)
    row = processed.iloc[45]
    day_index = int(
        frame.index[frame["date"] == row["date"]][0]
    )
    sales_history = frame["sales_volume"].tolist()[
        :day_index
    ]

    built = build_history_features(
        sales_history, as_of=row["date"]
    )

    for column in (
        "month",
        "dayofweek",
        "is_weekend",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
    ):
        assert built[column] == pytest.approx(
            float(row[column]), rel=1e-9, abs=1e-9
        ), f"Parity mismatch on {column}"


def test_history_builder_rejects_short_history():
    with pytest.raises(ValueError):
        build_history_features([100.0] * 10)
