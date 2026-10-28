"""
===============================================================================
PharmaSupply-ML: Master MLOps Pipeline Orchestrator
===============================================================================

OBJECTIVE & USAGE:
------------------
This master script orchestrates the end-to-end MLOps automation pipeline:
1. MOCK / Test Configuration & Cleanup: Optional development reset block ensuring
   a pristine environment by wiping obsolete local run artifacts before execution.
2. Infrastructure Validation: Ensures Docker, PostgreSQL (port 5432), and
   FastAPI (port 8000) environments are active and reachable.
3. Ingestion Automation: Systematically regenerates synthetic records from scratch
   (including custom batch sizes, test anomalies, critical expiry tracking for FEFO,
   and missing values) and ingests them into the PostgreSQL database.
4. Autonomous Data Validation & Cleaning: Passes ingested data through the
   Data Quality Gate (blocking fatal structural errors, auto-correcting minor anomalies
   and critical expiry constraints).
5. Feature Engineering: Extracts cleaned data and constructs chronological lag,
   rolling mean, and calendar variables.
6. Model Training: Trains the XGBoost demand forecasting model and persists
   the serialized artifact.
7. Explainability & Weight Export: Generates global/local SHAP interpretability
   plots and exports normalized feature weights for production monitoring.
8. Drift Monitoring: Executes Evidently AI data drift analysis using the
   weighted feature criticality score (`STABLE`, `WARNING`, `CRITICAL`).
9. REST API & Alerts Verification: Tests the FastAPI `/predict` and `/alerts` endpoints
   with full logistical and expiry payloads to confirm runtime integrity.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import json
import logging
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.data.generate_data import build_pharma_dataset
from src.explainability import ModelExplainer
from src.ingestion import ingest_csv_to_postgres
from src.monitoring import run_drift_monitoring
from src.train import (
    create_features,
    train_xgboost,
)
from src.validation import run_validation_and_clean

# Configure standard logging output for pipeline execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# MOCK BEGIN: Test Configuration & Mode
# =============================================================================
MOCK_RESET_MODE = True
# =============================================================================
# MOCK END
# =============================================================================


def is_port_open(
    host: str, port: int, timeout: float = 2.0
) -> bool:
    """Utility function to check socket connection on a target port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    is_open = sock.connect_ex((host, port)) == 0
    sock.close()
    return is_open


def ensure_docker_running() -> None:
    """Verifies PostgreSQL (5432) and FastAPI (8000) availability and attempts auto-start if down."""
    logging.info(
        "Step 0: Checking Docker infrastructure availability..."
    )

    pg_online = is_port_open("127.0.0.1", 5432)
    api_online = is_port_open("127.0.0.1", 8000)

    if pg_online and api_online:
        logging.info(
            "PostgreSQL (5432) and FastAPI (8000) services are online."
        )
        return

    logging.info(
        "One or more services unreachable. Attempting to start Docker Compose..."
    )
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"], check=True
        )

        for _ in range(15):
            if is_port_open("127.0.0.1", 8000):
                logging.info(
                    "Docker Compose services started and API is responsive."
                )
                return
            time.sleep(1)

    except Exception as e:
        logging.warning(
            f"Could not trigger Docker automatically: {e}. Ensure Docker Desktop is running."
        )


def reload_api_service() -> None:
    """Restarts the API container so it reloads the newly trained model joblib file."""
    logging.info(
        "Restarting FastAPI service to load updated model artifact..."
    )
    try:
        subprocess.run(
            ["docker", "compose", "restart", "api"],
            check=True,
        )
        time.sleep(2)
        logging.info(
            "FastAPI service successfully restarted."
        )
    except Exception as e:
        logging.warning(
            f"Could not restart API container automatically: {e}"
        )


def verify_api_serving() -> None:
    """Tests the API /predict and /alerts endpoints to confirm runtime integrity with FEFO constraints."""
    logging.info(
        "Step 6: Verifying REST API serving inference (/predict) and audit alerts (/alerts)..."
    )

    url_predict = "http://localhost:8000/predict"
    url_alerts = "http://localhost:8000/stores/1/alerts"

    payload = {
        "store_id": "1",
        "product_id": "MED_001",
        "stock_on_hand": 966.0,
        "supplier_lead_time": 2.0,
        "safety_threshold": 150.0,
        "batch_size": 50,
        "moq": 100,
        "shelf_life_days": 365,
        "lag_1": 96.0,
        "lag_7": 631.0,
        "lag_14": 1299.0,
        "lag_30": 2880.0,
    }

    try:
        # 1. Test Prediction Endpoint
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url_predict,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(
            req, timeout=5
        ) as response:
            if response.status == 200:
                result = json.loads(
                    response.read().decode("utf-8")
                )
                logging.info(
                    f"✅ API Predict Response Received: Predicted Demand = {result.get('predicted_demand'):.2f} | "
                    f"Reorder Needed = {result.get('reorder_needed')} | Model Status = {result.get('model_status')}"
                )
            else:
                logging.error(
                    f"❌ API Predict return status code: {response.status}"
                )

        # 2. Test Alerts Endpoint (Audit & FEFO)
        with urllib.request.urlopen(
            url_alerts, timeout=5
        ) as response_alerts:
            if response_alerts.status == 200:
                alerts_res = json.loads(
                    response_alerts.read().decode("utf-8")
                )
                logging.info(
                    f"✅ API Alerts Response Received: Store {alerts_res.get('store_id')} | "
                    f"Total active alerts = {len(alerts_res.get('alerts', []))}"
                )
                logging.info(
                    "✅ Step 6 SUCCESS: REST API serving and alerts validation completed successfully."
                )
            else:
                logging.error(
                    f"❌ API Alerts return status code: {response_alerts.status}"
                )

    except Exception as e:
        logging.error(f"❌ API Serving test failed: {e}")


def main() -> None:
    """Executes the full end-to-end MLOps pipeline sequentially with success tracing."""
    logging.info(
        "=========================================="
    )
    logging.info("STARTING PHARMASUPPLY MLOPS PIPELINE")
    logging.info(
        "=========================================="
    )

    if MOCK_RESET_MODE:
        logging.info(
            "🧹 [MOCK MODE] Cleaning previous local run artifacts to ensure a pristine start..."
        )
        obsolete_artifacts = [
            "model_status.txt",
            "reports/feature_weights.json",
            "reports/drift_report.html",
            "models/xgboost_pharma_demand.joblib",
        ]
        for artifact in obsolete_artifacts:
            p = Path(artifact)
            if p.exists():
                p.unlink()
                logging.info(
                    f"   - Removed obsolete artifact: {artifact}"
                )

    # 0. Infrastructure Check & Startup
    ensure_docker_running()
    logging.info(
        "✅ Step 0 SUCCESS: Infrastructure check & startup completed."
    )

    # 1. Automated Data Ingestion & Systematic Regeneration (Including Critical Expiry / FEFO)
    logging.info(
        "Step 1: Regenerating and ingesting raw data (with critical expiry tracking) into PostgreSQL..."
    )
    raw_data_path = Path(
        "data/pharmaceutical_demand_row.csv"
    )

    # Systematic regeneration from scratch incorporating critical_expiry_days
    df_pharma = build_pharma_dataset()
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_pharma.to_csv(raw_data_path, index=False)

    ingest_csv_to_postgres(
        file_path=raw_data_path,
        table_name="raw_pharmaceutical_demand",
    )
    logging.info(
        "✅ Step 1 SUCCESS: Raw dataset with FEFO parameters successfully regenerated and ingested into PostgreSQL."
    )

    # 2. Autonomous Data Quality Gate & Feature Engineering
    logging.info(
        "Step 2: Running Autonomous Data Quality Gate & engineering features..."
    )
    df_raw = run_validation_and_clean()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    df_processed = create_features(df_raw)
    logging.info(
        f"Dataset successfully validated and prepared. Total rows: {len(df_processed)}"
    )
    logging.info(
        "✅ Step 2 SUCCESS: Data Quality Gate and feature engineering completed."
    )

    # 3. Model Training & Persistence
    logging.info("Step 3: Training XGBoost model...")
    model, metrics, X_test, y_test = train_xgboost(
        df_processed
    )
    logging.info(
        f"Model trained successfully. Test MAE: {metrics['mae']:.4f}"
    )

    reload_api_service()
    logging.info(
        "✅ Step 3 SUCCESS: XGBoost model trained, evaluated, and deployed to memory."
    )

    # 4. Explainability (SHAP & Dynamic Weight Export)
    logging.info(
        "Step 4: Generating SHAP feature importance report and exporting weights..."
    )
    explainer = ModelExplainer(
        model_path="models/xgboost_pharma_demand.joblib",
        output_dir="reports/figures",
    )
    explainer.fit_explainer(
        X_test, threshold=300, sample_size=200, n_repeats=1
    )
    explainer.generate_global_plots(max_display=10)
    explainer.generate_local_breakdown(
        sample_index=0, max_display=10
    )
    explainer.export_shap_to_csv()
    explainer.export_feature_weights()
    logging.info(
        "✅ Step 4 SUCCESS: SHAP global/local plots and feature weights JSON successfully generated."
    )

    # 5. Weighted Data Drift Monitoring (Evidently AI)
    logging.info(
        "Step 5: Running Evidently AI weighted drift monitoring..."
    )
    run_drift_monitoring(
        df=df_processed,
        output_html_path="reports/drift_report.html",
    )
    logging.info(
        "✅ Step 5 SUCCESS: Weighted drift monitoring executed and HTML report generated."
    )

    # 6. REST API Inference & Alerts Validation
    verify_api_serving()

    logging.info(
        "=========================================="
    )
    logging.info("✅ PIPELINE EXECUTED SUCCESSFULLY!")
    logging.info("Generated Artifacts:")
    logging.info(
        " - Model artifact      : models/xgboost_pharma_demand.joblib"
    )
    logging.info(
        " - Feature weights     : reports/feature_weights.json"
    )
    logging.info(
        " - SHAP figures        : reports/figures/"
    )
    logging.info(
        " - Drift report        : reports/drift_report.html"
    )
    logging.info(
        " - API Prediction Endpoint : http://localhost:8000/predict"
    )
    logging.info(
        " - API Alerts Endpoint     : http://localhost:8000/stores/1/alerts"
    )
    logging.info(
        "=========================================="
    )


if __name__ == "__main__":
    main()
