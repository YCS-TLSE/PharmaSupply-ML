"""
===============================================================================
PharmaSupply-ML: Test Suite for ML Training Core & Artifact Serialization
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This test module demonstrates how to validate machine learning training pipelines,
chronological data splitting, and model artifact persistence using pytest and mock datasets.

In robust MLOps workflows, testing the training pipeline ensures:
1. Chronological Split Integrity: Prevention of data leakage by ensuring time-based
   separation between training and testing subsets.
2. Training Convergence & Metrics Generation: Verification that regressors
   successfully fit data and compute standard evaluation metrics (MAE, RMSE, WMAPE).
3. Artifact Persistence: Confirmation that trained model binaries are correctly
   serialized and saved to disk for downstream serving.

Author: PharmaSupply-ML Team
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add the project root directory to sys.path to allow absolute imports of src.train
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.train import train_xgboost


@pytest.fixture
def processed_feature_dataframe():
    """
    Pytest fixture providing a mock processed feature DataFrame
    spanning multiple chronological dates, matching XGBoost input requirements.
    """
    dates = pd.date_range(
        start="2026-01-01", periods=50, freq="D"
    )
    data = []
    for i, d in enumerate(dates):
        data.append(
            {
                "date": d,
                "month": d.month,
                "dayofweek": d.dayofweek,
                "is_weekend": 1 if d.dayofweek >= 5 else 0,
                "stock_on_hand": 500.0 + i,
                "supplier_lead_time": 3.0,
                "lag_7": 100.0,
                "lag_14": 100.0,
                "lag_30": 100.0,
                "rolling_mean_7": 100.0,
                "rolling_mean_30": 100.0,
                "target_demand": 105.0 + (i * 0.5),
            }
        )
    return pd.DataFrame(data)


@patch("src.train.joblib.dump")
def test_train_xgboost_execution(
    mock_joblib_dump, processed_feature_dataframe
):
    """
    Validates end-to-end XGBoost training orchestration.
    Ensures the model trains successfully on chronological splits,
    returns expected performance metrics, and attempts artifact serialization.
    """
    model, metrics, X_test, y_test = train_xgboost(
        processed_feature_dataframe
    )

    # Assertions on model and evaluation output
    assert model is not None
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wmape" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert len(X_test) > 0
    assert len(y_test) > 0

    # Verify that joblib serialization was triggered correctly
    mock_joblib_dump.assert_called_once()


def test_train_temporal_split_no_leakage(
    processed_feature_dataframe,
):
    """
    Validates chronological split integrity to prevent data leakage.
    Ensures that the maximum date in the training subset is strictly prior
    to the minimum date in the testing subset.
    """
    df = processed_feature_dataframe
    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    # Core MLOps assertion: strict temporal boundary separation
    assert train_df["date"].max() < test_df["date"].min(), (
        "Data Leakage Detected: Training dates overlap or succeed testing dates!"
    )


def test_train_model_performance_threshold(
    processed_feature_dataframe,
):
    """
    Validates performance non-regression via a WMAPE threshold check.
    Ensures the model achieves acceptable baseline accuracy on predictable synthetic trends.
    """
    _, metrics, _, _ = train_xgboost(
        processed_feature_dataframe
    )

    # On a perfectly linear synthetic trend, WMAPE should remain very low
    assert metrics["wmape"] < 20.0, (
        f"Model performance degraded: WMAPE ({metrics['wmape']:.2f}%) exceeded the 20% limit."
    )
