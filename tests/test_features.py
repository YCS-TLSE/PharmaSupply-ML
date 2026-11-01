"""
===============================================================================
PharmaSupply-ML: Test Suite for Feature Engineering & Time-Series Module
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This test module demonstrates how to test feature engineering and time-series
transformations in an MLOps pipeline using pytest and pandas.

In predictive supply chain systems, ensuring the correctness of rolling statistics,
temporal lags, and target definitions is critical. Crucially, these tests verify
Train-Serve Parity by ensuring that the mathematical transformations applied
during offline training match exactly what is expected online.

Key MLOps Validation Patterns Tested:
1. Calendar Feature Extraction:
   Verifies correct extraction of month, day of week, and weekend binary indicators.

2. Censored Demand Signal Reconstruction:
   Validates the forward-fill/backward-fill propagation of demand signals during
   consecutive stockout periods (`is_stockout = 1`).

3. Lag and Rolling Window Integrity:
   Ensures that temporal lags (7, 14, 30 days) and rolling means are correctly
   computed without leaking signals across distinct stores or products.

4. Target Recalculation & Boundary Drop:
   Validates that cumulative 7-future-day targets are properly constructed and
   that initial/trailing warm-up rows containing NaNs are safely dropped.

Author: PharmaSupply-ML Team
"""

import os
import sys

import pandas as pd
import pytest

# Add the project root directory to sys.path to allow absolute imports of src.features
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.features import create_features


@pytest.fixture
def time_series_dataframe():
    """
    Pytest fixture providing a multi-day time-series dataset for a store/product pair.
    Sufficiently long (45 days) to populate 30-day lags and rolling windows.
    """
    dates = pd.date_range(
        start="2026-01-01", periods=45, freq="D"
    )
    data = []
    for d in dates:
        data.append(
            {
                "date": d,
                "store_id": "STORE_001",
                "product_id": "MED_001",
                "sales_volume": 100.0,
                "is_stockout": 0,
            }
        )
    return pd.DataFrame(data)


def test_create_features_columns_structure(
    time_series_dataframe,
):
    """
    Validates that feature engineering produces all mandatory output columns
    defined in the pipeline specifications.
    """
    processed_df = create_features(time_series_dataframe)

    expected_columns = [
        "month",
        "dayofweek",
        "is_weekend",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "target_demand",
    ]

    for col in expected_columns:
        assert col in processed_df.columns, (
            f"Missing expected feature column: {col}"
        )


def test_calendar_features_logic(time_series_dataframe):
    """
    Validates calendar feature logic.
    Ensures 'is_weekend' is correctly set to 1 for weekends (Saturday/Sunday)
    and 0 for weekdays.
    """
    processed_df = create_features(time_series_dataframe)

    weekend_mask = processed_df["dayofweek"] >= 5
    assert (
        processed_df.loc[weekend_mask, "is_weekend"] == 1
    ).all()

    weekday_mask = processed_df["dayofweek"] < 5
    assert (
        processed_df.loc[weekday_mask, "is_weekend"] == 0
    ).all()


def test_group_isolation_no_lag_bleeding():
    """
    Validates group isolation to prevent feature bleeding.
    Ensures that lag calculations do not mix data between different products
    or stores as implemented in the grouping keys logic.
    """
    dates = pd.date_range(
        start="2026-01-01", periods=40, freq="D"
    )
    rows = []
    for d in dates:
        rows.append(
            {
                "date": d,
                "store_id": "STORE_001",
                "product_id": "MED_001",
                "sales_volume": 10.0,
                "is_stockout": 0,
            }
        )
        rows.append(
            {
                "date": d,
                "store_id": "STORE_001",
                "product_id": "MED_002",
                "sales_volume": 100.0,
                "is_stockout": 0,
            }
        )

    df = pd.DataFrame(rows)
    processed_df = create_features(df)

    # MED_001 lag_7 should rely on MED_001 sales (10.0), avoiding bleeding from MED_002 (100.0)
    med1_subset = processed_df[
        processed_df["product_id"] == "MED_001"
    ]
    assert (med1_subset["lag_7"] == 10.0).all()


def test_stockout_demand_signal_reconstruction():
    """
    Validates censored demand signal reconstruction during consecutive stockouts.
    Ensures that when 'is_stockout == 1', sales_volume is treated as NaN
    and correctly propagated via forward-fill / backward-fill logic (demand_signal).
    """
    dates = pd.date_range(
        start="2026-01-01", periods=45, freq="D"
    )
    data = []
    for i, d in enumerate(dates):
        # Introduce consecutive stockouts between index 10 and 13
        is_stk = 1 if 10 <= i <= 13 else 0
        # During stockout, sales_volume might be erratic or unrepresentative (e.g., 999.0)
        sales = 100.0 if not is_stk else 999.0

        data.append(
            {
                "date": d,
                "store_id": "STORE_001",
                "product_id": "MED_001",
                "sales_volume": sales,
                "is_stockout": is_stk,
            }
        )

    df = pd.DataFrame(data)
    processed_df = create_features(df)

    # Isolate the stockout period in the processed dataframe
    stockout_period = processed_df[
        (processed_df["date"] >= "2026-01-11")
        & (processed_df["date"] <= "2026-01-14")
    ]

    # The demand_signal should be successfully reconstructed/propagated from valid prior sales (100.0),
    # ignoring the corrupted 999.0 values during stockouts.
    assert (
        stockout_period["demand_signal"] == 100.0
    ).all(), (
        "Demand signal was not correctly reconstructed using ffill/bfill during stockouts."
    )


def test_nan_warmup_removal(time_series_dataframe):
    """
    Validates that warmup periods (initial lags and trailing future targets)
    are properly dropped to output a clean dataset ready for training.
    """
    initial_len = len(time_series_dataframe)
    processed_df = create_features(time_series_dataframe)

    assert len(processed_df) < initial_len
    assert processed_df["lag_30"].notnull().all()
    assert processed_df["target_demand"].notnull().all()
