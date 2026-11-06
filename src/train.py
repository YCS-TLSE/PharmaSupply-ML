"""
===============================================================================
PharmaSupply-ML: Temporal Split & XGBoost Training Core
===============================================================================

PURPOSE
-------
Train the demand forecasting model with a leakage-free chronological split and
persist a reproducible artifact for the serving layer.

RESPONSIBILITIES
----------------
* Load data through the autonomous validation/cleaning quality gate.
* Split observations by date (80/20) so the test set is strictly in the future.
* Fit an XGBoost regressor on the shared feature contract.
* Compute MAE, RMSE and WMAPE for regression monitoring.
* Persist the joblib artifact and optionally log params, metrics, signature and
  model to MLflow (tracking + registry).

INPUTS / OUTPUTS
----------------
Inputs: a validated DataFrame with dates, features and ``TARGET_COL``.
Outputs: the fitted model, a metrics dictionary, the test matrices and the
serialised artifact at ``MODEL_PATH``.

ARCHITECTURE INTEGRATION
------------------------
Called by ``run_pipeline.py`` and ``src.monitoring``. MLflow logging is
best-effort and disabled by default, so training never depends on an external
tracking server.
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

# Shared feature contract, validation gate and feature engineering
from src.config import (
    FEATURE_COLS,
    MLFLOW_ARTIFACT_PATH,
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    TARGET_COL,
)
from src.features import create_features
from src.validation import run_validation_and_clean

# Load environment variables from .env file
load_dotenv()

# Configure standard logging output for MLOps monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Optional MLflow bridge. Tracking failures are logged and swallowed because a
# registry outage must never prevent the model artifact from being produced.
def _log_to_mlflow(
    model,
    metrics: dict[str, float],
    params: dict[str, float],
    X_train: pd.DataFrame,
) -> str | None:
    """Log params, metrics and a signed model to MLflow.

    Tracking is best-effort: a registry outage must never fail training.
    """
    if not MLFLOW_ENABLED:
        return None

    try:
        import mlflow
        import mlflow.xgboost
        from mlflow.models import infer_signature
    except ImportError:
        logging.warning(
            "MLflow is not installed; tracking skipped."
        )
        return None

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            sample = X_train.head(5)
            signature = infer_signature(
                sample, model.predict(sample)
            )
            mlflow.xgboost.log_model(
                model,
                artifact_path=MLFLOW_ARTIFACT_PATH,
                signature=signature,
                input_example=sample,
                registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            )
            return run.info.run_id
    except (
        Exception
    ) as exc:  # pragma: no cover - external service
        logging.warning("MLflow logging failed: %s", exc)
        return None


# Data access entry point for the training pipeline. Validation and cleaning
# happen upstream so training only sees structurally valid rows.
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


# Core training routine: chronological split, fit, evaluation, persistence and
# optional MLflow publication. The split is performed on unique dates, not on
# shuffled rows, to avoid temporal leakage.
def train_xgboost(df: pd.DataFrame):
    """Train an XGBoost regressor using a time-based split and evaluate performance."""
    feature_cols = list(FEATURE_COLS)
    target_col = TARGET_COL

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

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    logging.info(
        f"Model artifact saved successfully -> {model_path}"
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "wmape": float(wmape),
    }
    params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 5,
        "base_score": 0.5,
        "random_state": 42,
        "feature_count": len(feature_cols),
    }
    run_id = _log_to_mlflow(model, metrics, params, X_train)
    if run_id:
        metrics["mlflow_run_id"] = run_id

    return (
        model,
        metrics,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    df_raw = load_validated_data()
    df_features = create_features(df_raw)
    train_xgboost(df_features)
