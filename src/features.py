"""
===============================================================================
PharmaSupply-ML: Feature Engineering & Time-Series Processing Module
===============================================================================

OBJECTIVE & USAGE:
------------------
This module extracts feature engineering logic from the training script to 
ensure absolute training/serving parity (avoiding train/serve skew) for the 
pharmaceutical supply chain forecasting system.
"""

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
    df["target_demand"] = df.groupby(group_cols)[
        "demand_signal"
    ].transform(
        lambda x: (
            x.rolling(7, min_periods=7).sum().shift(-6)
        )
    )

    # Drop initial NaN rows (lags) and ending NaN rows (future target window)
    df = df.dropna(
        subset=["lag_30", "target_demand"]
    ).reset_index(drop=True)

    return df