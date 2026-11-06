"""
===============================================================================
PharmaSupply-ML: Master MLOps Pipeline Orchestrator
===============================================================================

PURPOSE
-------
Run the complete local MLOps lifecycle in a deterministic, fail-fast order:
infrastructure check, data generation/ingestion, validation, feature
engineering, training, explainability, drift monitoring and API smoke tests.

RESPONSIBILITIES
----------------
* Verify Docker, PostgreSQL and the API; start compose when needed.
* Regenerate and upsert the synthetic dataset.
* Run the data quality gate and feature engineering.
* Train the model, rebuild the API image and wait for readiness.
* Generate SHAP artifacts and the offline drift report.
* Exercise `/predict` and `/alerts` with retries to survive startup windows.

INPUTS / OUTPUTS
----------------
Inputs: repository files, Docker compose, environment variables.
Outputs: PostgreSQL rows, model artifact, SHAP figures, drift report, model
status and a successful API smoke-test exit code.

ARCHITECTURE INTEGRATION
------------------------
This is the developer/CI entry point that wires every module together. The
API is rebuilt (not merely restarted) so new Python dependencies are always
present, and readiness is confirmed through ``/readyz`` before smoke tests.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import argparse
import json
import logging
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from scripts.clean_artifacts import clean_artifacts
from src.config import API_KEY, API_KEY_HEADER
from src.data.generate_data import build_pharma_dataset
from src.explainability import ModelExplainer
from src.features import create_features
from src.ingestion import ingest_csv_to_postgres
from src.monitoring import run_drift_monitoring
from src.train import train_xgboost
from src.validation import run_validation_and_clean

# Configure standard logging output for pipeline execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Shared authentication header for readiness and smoke-test requests.
def _auth_headers() -> dict:
    """Return the API-key header when authentication is enabled."""
    return {API_KEY_HEADER: API_KEY} if API_KEY else {}


# Low-level socket probe used before starting compose. It checks reachability,
# not application readiness (see wait_for_api_ready).
def is_port_open(
    host: str, port: int, timeout: float = 2.0
) -> bool:
    """Utility function to check socket connection on a target port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    is_open = sock.connect_ex((host, port)) == 0
    sock.close()
    return is_open


# Poll /readyz until the model is loaded and PostgreSQL is reachable. This
# prevents smoke tests from hitting a container that is still starting.
def wait_for_api_ready(timeout: int = 90) -> bool:
    """Wait until the API readiness probe reports model + database ready."""
    deadline = time.monotonic() + timeout
    url = "http://localhost:8000/readyz"

    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                url, headers=_auth_headers()
            )
            with urllib.request.urlopen(
                request, timeout=3
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)

    return False


# Retry transient network failures and 5xx responses. 4xx responses are real
# client errors and fail immediately.
def _fetch_with_retry(
    make_request,
    timeout: int = 10,
    attempts: int = 6,
    delay: float = 2.0,
) -> tuple[int, str]:
    """Execute an HTTP request, retrying transient connection/5xx errors."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                make_request(), timeout=timeout
            ) as response:
                return (
                    response.status,
                    response.read().decode("utf-8"),
                )
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last_error = exc
        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc

        logging.warning(
            "API request attempt %d/%d failed: %s",
            attempt,
            attempts,
            last_error,
        )
        if attempt < attempts:
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        "API request failed without a captured error."
    )


# Step 0: best-effort infrastructure startup. The API image itself is rebuilt
# later, once the new model artifact has been trained.
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
        # Prevent Docker from mounting a non-existent file as a directory
        status_path = Path("model_status.txt")
        if status_path.is_dir():
            shutil.rmtree(status_path)
        if not status_path.exists():
            status_path.write_text("STABLE")

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
        logging.error(
            f"Failed to initialize Docker infrastructure: {e}"
        )
        sys.exit(1)


# Step 3 helper: rebuild the API image so code and dependencies match the
# repository, then wait for readiness before continuing.
def reload_api_service() -> None:
    """Rebuild and recreate the API container, then wait for readiness.

    A plain ``docker compose restart`` reuses the existing image and therefore
    cannot pick up new Python dependencies (for example SQLAlchemy introduced
    by the repository layer). Rebuilding guarantees that the running code and
    its dependencies match the repository state.
    """
    logging.info(
        "Rebuilding and restarting FastAPI service (code + dependencies)..."
    )
    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "up",
                "-d",
                "--build",
                "api",
            ],
            check=True,
        )
    except Exception as e:
        logging.error(
            f"Could not rebuild API container automatically: {e}"
        )
        raise RuntimeError(
            "API image rebuild failed. Check `docker compose logs api`."
        ) from e

    if not wait_for_api_ready(timeout=120):
        raise RuntimeError(
            "FastAPI did not become ready within 120s after rebuild. "
            "Check `docker compose logs api`."
        )

    logging.info(
        "FastAPI service successfully rebuilt and ready."
    )


# Step 6: validate inference and alerts through the public HTTP contract.
def verify_api_serving() -> None:
    """Tests /predict and /alerts, retrying through API startup/reset windows."""
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
        "lag_7": 96.0,
        "lag_14": 96.0,
        "lag_30": 96.0,
        # Exercise the shared train/serve feature builder.
        "history": [96.0] * 30,
    }
    data = json.dumps(payload).encode("utf-8")

    try:
        # 1. Test Prediction Endpoint
        status_code, body = _fetch_with_retry(
            lambda: urllib.request.Request(
                url_predict,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    **_auth_headers(),
                },
            )
        )
        if status_code != 200:
            logging.error(
                "API Predict returned non-200 status code: %s",
                status_code,
            )
            sys.exit(1)

        result = json.loads(body)
        logging.info(
            f"API Predict Response Received: Predicted Demand = {result.get('predicted_demand'):.2f} | "
            f"Reorder Needed = {result.get('reorder_needed')} | Model Status = {result.get('model_status')}"
        )

        # 2. Test Alerts Endpoint (Audit & FEFO)
        status_alerts, body_alerts = _fetch_with_retry(
            lambda: urllib.request.Request(
                url_alerts, headers=_auth_headers()
            )
        )
        if status_alerts != 200:
            logging.error(
                "API Alerts returned non-200 status code: %s",
                status_alerts,
            )
            sys.exit(1)

        alerts_res = json.loads(body_alerts)
        logging.info(
            f"API Alerts Response Received: Store {alerts_res.get('store_id')} | "
            f"Total active alerts = {len(alerts_res.get('alerts', []))}"
        )
        logging.info(
            "Step 6 SUCCESS: REST API serving and alerts validation completed successfully."
        )

    except Exception as e:
        logging.error(f"API Serving test failed: {e}")
        sys.exit(1)


# Sequential orchestrator with fail-fast semantics. Each step logs its outcome
# and exits with a non-zero status on failure so CI can block a bad release.
def main(clean_first: bool = False) -> None:
    """Executes the full end-to-end MLOps pipeline sequentially with success tracing and fail-fast handling."""
    logging.info(
        "=========================================="
    )
    logging.info("STARTING PHARMASUPPLY MLOPS PIPELINE")
    logging.info(
        "=========================================="
    )

    if clean_first:
        logging.info(
            "Pre-run cleanup requested: delegating to scripts.clean_artifacts..."
        )
        clean_artifacts(
            reset_status=True,
            include_figures=False,
            verbose=True,
        )

    # Step 0: Infrastructure Check & Startup
    try:
        ensure_docker_running()
        logging.info(
            "Step 0 SUCCESS: Infrastructure check & startup completed."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 0: {e}")
        sys.exit(1)

    # Step 1: Automated Data Ingestion & Systematic Regeneration
    try:
        logging.info(
            "Step 1: Regenerating and ingesting raw data (with critical expiry tracking) into PostgreSQL..."
        )
        raw_data_path = Path(
            "data/pharmaceutical_demand_row.csv"
        )

        df_pharma = build_pharma_dataset()
        raw_data_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        df_pharma.to_csv(raw_data_path, index=False)

        ingest_csv_to_postgres(
            file_path=raw_data_path,
            table_name="raw_pharmaceutical_demand",
        )
        logging.info(
            "Step 1 SUCCESS: Raw dataset with FEFO parameters successfully regenerated and ingested into PostgreSQL."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 1: {e}")
        sys.exit(1)

    # Step 2: Autonomous Data Quality Gate & Feature Engineering
    try:
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
            "Step 2 SUCCESS: Data Quality Gate and feature engineering completed."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 2: {e}")
        sys.exit(1)

    # Step 3: Model Training & Persistence
    try:
        logging.info("Step 3: Training XGBoost model...")
        model, metrics, X_test, y_test = train_xgboost(
            df_processed
        )
        logging.info(
            f"Model trained successfully. Test MAE: {metrics['mae']:.4f}"
        )

        reload_api_service()
        logging.info(
            "Step 3 SUCCESS: XGBoost model trained, evaluated, and deployed to memory."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 3: {e}")
        sys.exit(1)

    # Step 4: Explainability (SHAP & Dynamic Weight Export)
    try:
        logging.info(
            "Step 4: Generating SHAP feature importance report and exporting weights..."
        )
        explainer = ModelExplainer(
            model_path="models/xgboost_pharma_demand.joblib",
            output_dir="reports/figures",
        )
        explainer.fit_explainer(
            X_test,
            threshold=300,
            sample_size=200,
            n_repeats=1,
        )
        explainer.generate_global_plots(max_display=10)
        explainer.generate_local_breakdown(
            sample_index=0, max_display=10
        )
        explainer.export_shap_to_csv()
        explainer.export_feature_weights()
        logging.info(
            "Step 4 SUCCESS: SHAP global/local plots and feature weights JSON successfully generated."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 4: {e}")
        sys.exit(1)

    # Step 5: Weighted Data Drift Monitoring (Evidently AI)
    try:
        logging.info(
            "Step 5: Running Evidently AI weighted drift monitoring..."
        )
        run_drift_monitoring(
            df=df_processed,
            output_html_path="reports/drift_report.html",
        )
        logging.info(
            "Step 5 SUCCESS: Weighted drift monitoring executed and HTML report generated."
        )
    except Exception as e:
        logging.error(f"Pipeline failed at Step 5: {e}")
        sys.exit(1)

    # Step 6: REST API Inference & Alerts Validation
    verify_api_serving()

    logging.info(
        "=========================================="
    )
    logging.info("PIPELINE EXECUTED SUCCESSFULLY!")
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
    parser = argparse.ArgumentParser(
        description="PharmaSupply-ML: Master MLOps Pipeline Orchestrator"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous run artifacts before executing the pipeline (calls scripts/clean_artifacts.py).",
    )
    args = parser.parse_args()
    main(clean_first=args.clean)
