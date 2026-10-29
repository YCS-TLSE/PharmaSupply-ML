"""
===============================================================================
PharmaSupply-ML: Temporal Split & XGBoost Training Core
===============================================================================

OBJECTIVE & USAGE:
------------------
This script forms the Machine Learning core of the MLOps pipeline. It:
1. Integrates with the autonomous data validation module (`validation.py`) to
   guarantee clean, auto-corrected input data.
2. Imports the decoupled feature engineering module (`features.py`).
3. Performs a rigorous time-based chronological split (preventing data leakage).
4. Trains an XGBoost regressor to forecast pharmaceutical product demand and
   persists the serialized artifact.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBRegressor

# Import the autonomous validation and cleaning gate
from src.validation import run_validation_and_clean
# Import decoupled feature engineering for Train/Serve parity
from src.features import create_features

# Load environment variables from .env file
load_dotenv()

# Configure standard logging output for MLOps monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_validated_data() -> pd.DataFrame:
    """Fetches raw data through the automated validation and cleaning quality gate,

    ensuring all anomalies are auto-corrected or structurally verified.
    """
    logging.info(
        "Initiating data loading via Autonomous Validation Quality Gate..."
    )
    df = run_validation_and_clean()

    # Cast date column to explicit datetime objects for temporal processing
    df["date"] = pd.to_datetime(df["date"])
    return df


def train_xgboost(df: pd.DataFrame):
    """Train an XGBoost regressor using a time-based split and evaluate performance."""
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

    # Instantiate XGBoost regressor with explicit base_score to prevent SHAP parsing crash
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        base_score=0.5,
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

    return (
        model,
        {"mae": mae, "rmse": rmse, "wmape": wmape},
        X_test,
        y_test,
    )


if __name__ == "__main__":
    df_raw = load_validated_data()
    df_features = create_features(df_raw)
    train_xgboost(df_features)