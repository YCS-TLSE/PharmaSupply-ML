"""Module for feature engineering, time-based split, and XGBoost model training.

CONTEXT & PURPOSE:
------------------
This script forms the Machine Learning core of the MLOps pipeline. It reads raw
ingested data from PostgreSQL, constructs temporal and lag features, and trains
an XGBoost regressor to forecast pharmaceutical product demand.

MODEL SELECTION (WHY XGBOOST?):
-------------------------------
1. Tabular Efficiency: Gradient Boosted Decision Trees consistently outperform
   deep learning models on structured tabular time-series data with small to
   medium observation counts.
2. Nonlinear & Seasonal Dynamics: Handles non-linear trends, complex multi-year
   seasonalities (winter flu, spring allergies), and sudden demand spikes without
   requiring strict data stationarity assumptions.
3. Feature Importance & Interpretability: Provides direct insights into key drivers
   (e.g., lag weight vs. stock levels), which is essential for supply chain planning.
4. Robustness to Noise & Outliers: Tree-based partitioning is resilient to delivery
   shocks and temporary supply chain anomalies.

DESIGN & ARCHITECTURE:
----------------------
1. Data-Agnostic Contract:
   The pipeline assumes no specific generator logic. It strictly relies on a
   standard schema (date, product_id, sales_volume, stock_on_hand, etc.), making it
   ready to process real-world client data.

2. Censored Demand Handling:
   To prevent zero-sales during stockouts from skewing lag features, stockout periods
   are masked and forward-filled (demand signal reconstruction).

3. Time-Based Validation:
   Prevents temporal data leakage by splitting train/test sets sequentially along
   the timeline rather than using random cross-validation.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from xgboost import XGBRegressor

# Load environment variables from .env file
load_dotenv()

# Configure standard logging output for MLOps monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Extract database credentials from environment variables
DB_USER = os.getenv("POSTGRES_USER", "ycs_admin")
DB_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD", "ycs_admin_secure_password"
)
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "pharmasupply_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_data_from_db() -> pd.DataFrame:
    """Fetch raw historical records from PostgreSQL, ordered chronologically.

    Returns:
        pd.DataFrame: Raw dataset containing daily stock and sales records.
    """
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM raw_pharmaceutical_demand ORDER BY date ASC;"

    logging.info(
        "Fetching raw data from PostgreSQL database..."
    )
    df = pd.read_sql(query, engine)

    # Cast date column to explicit datetime objects for temporal processing
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer calendar, lag, and rolling window features for time-series forecasting.

    Args:
        df (pd.DataFrame): Raw transactional data sorted by product and date.

    Returns:
        pd.DataFrame: Enriched dataset containing lag features and target variables.
    """
    logging.info(
        "Engineering dynamic features (calendar, lags, rolling statistics)..."
    )

    # Ensure strict sorting by product and time before calculating rolling stats
    df = df.sort_values(["product_id", "date"]).reset_index(
        drop=True
    )

    # --- 1. Calendar Features ---
    # Extract seasonal patterns directly from the date object
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- 2. Censored Demand Reconstruction ---
    # When a stockout occurs (is_stockout = 1), sales_volume drops to 0 or stock limit.
    # We replace stockout sales with NaN to avoid propagating false "zero-demand" into lags.
    df["demand_signal"] = np.where(
        df["is_stockout"] == 1, np.nan, df["sales_volume"]
    )

    # Impute missing demand signals during stockouts via forward/backward filling per product
    df["demand_signal"] = (
        df.groupby("product_id")["demand_signal"]
        .ffill()
        .bfill()
    )

    # --- 3. Lag and Rolling Window Features ---
    # Create historical lags (7, 14, and 30 days) per product group
    for lag in [7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby("product_id")[
            "demand_signal"
        ].shift(lag)

    # Create rolling averages over past 7 and 30 days (shift by 1 to prevent data leakage)
    for window in [7, 30]:
        df[f"rolling_mean_{window}"] = df.groupby(
            "product_id"
        )["demand_signal"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    # Drop initial rows per product where lag values are NaN due to shifting
    df = df.dropna().reset_index(drop=True)
    return df


def train_xgboost(df: pd.DataFrame):
    """Train an XGBoost regressor using a time-based split and evaluate performance.

    Args:
        df (pd.DataFrame): Enriched dataset containing target and engineered features.

    Returns:
        tuple: (model, metrics, X_test, y_test)
    """
    feature_cols = [
        "month",
        "dayofweek",
        "is_weekend",
        "stock_on_hand",
        "supplier_lead_time",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
    ]
    target_col = "target_demand"

    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_train, y_train = (
        train_df[feature_cols],
        train_df[target_col],
    )
    X_test, y_test = (
        test_df[feature_cols],
        test_df[target_col],
    )

    logging.info(
        f"Train timeline: {train_df['date'].min().strftime('%Y-%m-%d')} to "
        f"{train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)"
    )
    logging.info(
        f"Test timeline:  {test_df['date'].min().strftime('%Y-%m-%d')} to "
        f"{test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)"
    )

    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    wmape = (
        np.sum(np.abs(y_test - predictions))
        / np.sum(y_test)
        * 100
    )

    metrics = {"mae": mae, "rmse": rmse, "wmape": wmape}

    logging.info("--- Model Evaluation Results ---")
    logging.info(f"MAE:   {mae:.2f}")
    logging.info(f"RMSE:  {rmse:.2f}")
    logging.info(f"WMAPE: {wmape:.2f}%")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "xgboost_pharma_demand.joblib"

    joblib.dump(model, model_path)
    logging.info(
        f"Model artifact saved successfully -> {model_path}"
    )

    return model, metrics, X_test, y_test


if __name__ == "__main__":
    df_raw = load_data_from_db()
    df_features = create_features(df_raw)
    train_xgboost(df_features)
