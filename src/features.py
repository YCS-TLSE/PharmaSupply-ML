"""
===============================================================================
PharmaSupply-ML: Feature Engineering & Time-Series Processing
===============================================================================

PURPOSE
-------
Provide the canonical feature transformations used by both training and
inference, eliminating train/serve skew.

RESPONSIBILITIES
----------------
* Build calendar features (month, day of week, weekend flag).
* Reconstruct censored demand during stockout periods.
* Compute temporal lags and rolling means per store/product group.
* Build the 7-day cumulative target used by the regressor.
* Expose ``build_history_features`` for next-day inference from a chronological
  sales history, using exactly the same formulas as ``create_features``.

INPUTS / OUTPUTS
----------------
Inputs: a raw/validated DataFrame with date, store, product, sales and stockout
columns, or a sequence of daily sales for inference.
Outputs: a feature-enriched DataFrame, or a dictionary with the model's
time-series feature vector.

ARCHITECTURE INTEGRATION
------------------------
Used by ``src.train``, ``src.monitoring``, ``src.explainability`` and
``src.services.inventory``. ``src.config.TARGET_COL`` is the single target
column name shared with the model contract.
"""

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Training-side feature builder. Grouping by store/product prevents lag
# bleeding across independent inventory series.
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer calendar, lag, and rolling window features, plus 7-day cumulative target."""
    logging.info(
        "Engineering dynamic features (calendar, lags, rolling statistics)..."
    )

    # Grouping keys to avoid bleeding lag signals across different stores/products
    group_cols = (
        ["store_id", "product_id"]
        if "store_id" in df.columns
        else ["product_id"]
    )

    # Ensure strict sorting by store, product, and time before calculating rolling stats
    sort_cols = group_cols + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 1. Calendar Features ---
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- 2. Censored Demand Reconstruction ---
    df["demand_signal"] = np.where(
        df["is_stockout"] == 1, np.nan, df["sales_volume"]
    )
    df["demand_signal"] = (
        df.groupby(group_cols)["demand_signal"]
        .ffill()
        .bfill()
    )

    # --- 3. Lag and Rolling Window Features ---
    for lag in [7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby(group_cols)[
            "demand_signal"
        ].shift(lag)

    for window in [7, 30]:
        df[f"rolling_mean_{window}"] = df.groupby(
            group_cols
        )["demand_signal"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    # --- 4. TARGET RECALCULATION (CUMULATED DEMAND OVER NEXT 7 DAYS) ---
    df[TARGET_COL] = df.groupby(group_cols)[
        "demand_signal"
    ].transform(
        lambda x: (
            x.rolling(7, min_periods=7).sum().shift(-6)
        )
    )

    # Drop initial NaN rows (lags) and ending NaN rows (future target window)
    df = df.dropna(
        subset=["lag_30", TARGET_COL]
    ).reset_index(drop=True)

    return df


# Inference-side feature builder. It mirrors create_features for the next day:
# lags are the values observed 7/14/30 days before the target day, and rolling
# means cover the N days preceding it. A parity test protects this contract.
def build_history_features(
    sales_history: Sequence[float],
    as_of: Optional[pd.Timestamp] = None,
    min_history: int = 30,
) -> dict[str, float]:
    """Build the model's time-series features for the next day.

    This is the single inference-side implementation of the formulas used by
    :func:`create_features`. Given a chronological history *excluding* the
    target day, it reproduces exactly:

    * ``lag_N`` = demand observed ``N`` days before the target day;
    * ``rolling_mean_N`` = mean of the ``N`` days preceding the target day.

    A parity test compares this function with ``create_features`` to prevent
    train/serve skew.
    """
    if len(sales_history) < min_history:
        raise ValueError(
            f"At least {min_history} daily sales values are required."
        )

    series = pd.Series(
        [float(value) for value in sales_history],
        dtype="float64",
    )
    reference = (
        pd.Timestamp(as_of)
        if as_of is not None
        else pd.Timestamp.now()
    )

    return {
        "month": float(reference.month),
        "dayofweek": float(reference.dayofweek),
        "is_weekend": float(reference.dayofweek >= 5),
        "lag_7": float(series.iloc[-7]),
        "lag_14": float(series.iloc[-14]),
        "lag_30": float(series.iloc[-30]),
        "rolling_mean_7": float(series.iloc[-7:].mean()),
        "rolling_mean_30": float(series.iloc[-30:].mean()),
    }
