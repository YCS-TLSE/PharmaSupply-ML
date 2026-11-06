"""
===============================================================================
PharmaSupply-ML: Central Configuration & Shared Constants
===============================================================================

PURPOSE
-------
Provide one importable source of truth for every value that must stay
consistent across data ingestion, validation, training, serving, monitoring
and the dashboard. Before this module existed, feature lists, default values,
status labels and filesystem paths were duplicated across modules, which made
silent divergence likely.

RESPONSIBILITIES
----------------
* Resolve project paths (model artifact, drift status file, prediction log).
* Declare the model feature contract consumed by training and inference.
* Centralise business defaults (safety threshold, MOQ, shelf life, etc.).
* Define operational rules (FEFO horizon, lead-time buffer, alert thresholds).
* Expose drift thresholds and the accepted model-status vocabulary.
* Read environment-driven settings for MLflow, security, observability and
  SMTP notifications.

INPUTS / OUTPUTS
----------------
* Inputs: process environment variables and an optional ``.env`` file located
  at the repository root.
* Outputs: module-level constants imported by every other package. This module
  performs no I/O beyond loading ``.env`` and creating ``Path`` objects; it has
  no side effects on the database, model or network.

ARCHITECTURE INTEGRATION
------------------------
Imported by ``src.api``, ``src.train``, ``src.monitoring``, ``src.ingestion``,
``src.validation``, ``src.dashboard``, ``src.security`` and ``src.observability``.
Because it imports nothing from the rest of the project, it is safe to import
from any layer without creating a cycle.

OPERATIONAL NOTES
-----------------
Environment variables override code defaults. In production, secrets
(database password, SMTP password, API key) must be injected by the runtime
platform, never committed. Boolean flags accept ``1/true/yes``.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# 1. PERSISTENCE PATHS
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(
        ROOT_DIR / "models" / "xgboost_pharma_demand.joblib"
    ),
)

STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH",
    str(ROOT_DIR / "model_status.txt"),
)

# =============================================================================
# 2. MODEL CONTRACT
# =============================================================================
# Order matters: the API builds its inference DataFrame with these columns.
FEATURE_COLS: tuple[str, ...] = (
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
)

TARGET_COL = "target_demand"

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# =============================================================================
# 3. BUSINESS DEFAULTS
# =============================================================================
DEFAULT_SAFETY_THRESHOLD = 150.0
DEFAULT_SUPPLIER_LEAD_TIME = 3.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_MOQ = 1
DEFAULT_SHELF_LIFE_DAYS = 365

# =============================================================================
# 4. OPERATIONAL RULES
# =============================================================================
# Level-1 alert: safety threshold below 1.5x target demand.
SAFETY_THRESHOLD_DEMAND_MULTIPLIER = 1.5
# Level-1 alert: excessive over-immobilisation threshold.
MAX_SAFETY_THRESHOLD = 3000.0
# Safety buffer applied on top of the supplier lead time.
LEAD_TIME_BUFFER_DAYS = 1.0
# Batches expiring within this horizon are considered for FEFO loss.
FEFO_HORIZON_DAYS = 7
# Level-2 alert: expiry warning threshold.
FEFO_CRITICAL_DAYS = 30

# =============================================================================
# 5. DRIFT MONITORING
# =============================================================================
DRIFT_WARNING_THRESHOLD = 0.15
DRIFT_CRITICAL_THRESHOLD = 0.30

# Values accepted from the monitoring batch in ``model_status.txt``.
MODEL_STATUS_VALUES = frozenset(
    {"STABLE", "WARNING", "CRITICAL"}
)

# =============================================================================
# 6. RUNTIME & NOTIFICATIONS
# =============================================================================
MODEL_STATUS_POLL_SECONDS = int(
    os.getenv("MODEL_STATUS_POLL_SECONDS", "30")
)
SMTP_TIMEOUT_SECONDS = int(
    os.getenv("SMTP_TIMEOUT_SECONDS", "5")
)

SMTP_SERVER_DEFAULT = os.getenv(
    "SMTP_SERVER", "smtp.gmail.com"
)
SMTP_PORT_DEFAULT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL_DEFAULT = os.getenv(
    "SENDER_EMAIL", "mlops-alert@pharmasupply.com"
)
RECEIVER_EMAIL_DEFAULT = os.getenv(
    "RECEIVER_EMAIL", "support-tech@pharmasupply.com"
)

# =============================================================================
# 7. OPERATOR-FACING MESSAGES (shared by API and dashboard)
# =============================================================================
HEALTH_MESSAGES: dict[str, str] = {
    "STABLE": "🟢 STABLE — Le modèle de prévision est fiable",
    "WARNING": (
        "🟠 WARNING — Performance du modèle sous surveillance : nous vous"
        " invitons à considérer les résultats ci-dessous avec une vigilance"
        " renforcée. Nos équipes optimisent le système."
    ),
    "CRITICAL": (
        "🔴 CRITICAL — Alerte critique sur le modèle de prévision, nous vous"
        " invitons à considérer les résultats ci-dessous avec une extrême"
        " vigilance, le support client vous contactera sous 24h."
    ),
    # Fail-safe state: never present an unreadable status as STABLE.
    "UNKNOWN": (
        "🟠 UNKNOWN — Statut de santé du modèle indisponible : les prévisions"
        " doivent être considérées avec prudence jusqu'au rétablissement du"
        " monitoring."
    ),
}

# =============================================================================
# 8. DATABASE POOL
# =============================================================================
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "5"))

# =============================================================================
# 9. MLFLOW TRACKING & MODEL REGISTRY
# =============================================================================
MLFLOW_ENABLED = os.getenv(
    "MLFLOW_ENABLED", "false"
).lower() in {
    "1",
    "true",
    "yes",
}
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "file:./mlruns"
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "pharmasupply-demand"
)
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "PharmaSupplyDemandForecaster",
)
MLFLOW_ARTIFACT_PATH = os.getenv(
    "MLFLOW_ARTIFACT_PATH", "model"
)

# =============================================================================
# 10. SECURITY (API KEY, CORS, RATE LIMITING)
# =============================================================================
# Empty API_KEY disables authentication (local development only).
API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:8501"
    ).split(",")
    if origin.strip()
)
RATE_LIMIT_PREDICT = os.getenv(
    "RATE_LIMIT_PREDICT", "60/minute"
)
RATE_LIMIT_ALERTS = os.getenv(
    "RATE_LIMIT_ALERTS", "120/minute"
)

# =============================================================================
# 11. OBSERVABILITY & PRODUCTION DRIFT
# =============================================================================
PREDICTION_LOG_PATH = os.getenv(
    "PREDICTION_LOG_PATH",
    str(ROOT_DIR / "reports" / "prediction_log.jsonl"),
)
PRODUCTION_DRIFT_MIN_SAMPLES = int(
    os.getenv("PRODUCTION_DRIFT_MIN_SAMPLES", "30")
)
FEATURE_WEIGHTS_PATH = os.getenv(
    "FEATURE_WEIGHTS_PATH",
    str(ROOT_DIR / "reports" / "feature_weights.json"),
)

# Fallback weights used only when reports/feature_weights.json is absent.
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "lag_7": 0.25,
    "stock_on_hand": 0.20,
    "lag_14": 0.15,
    "lag_30": 0.10,
    "rolling_mean_7": 0.10,
    "rolling_mean_30": 0.10,
    "supplier_lead_time": 0.05,
    "month": 0.02,
    "dayofweek": 0.02,
    "is_weekend": 0.01,
}
