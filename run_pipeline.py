"""Master MLOps Pipeline Orchestrator.

Sequentially executes:
1. Docker infrastructure check/startup (Postgres + FastAPI).
2. Raw data generation (if missing) & PostgreSQL ingestion.
3. Feature engineering & data extraction from DB.
4. Model training & evaluation.
5. SHAP Explainability report.
6. Evidently AI Drift Monitoring.
7. REST API Integration & Serving test (/predict).
"""

import json
import logging
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from src.explainability import ModelExplainer
from src.ingestion import ingest_csv_to_postgres
from src.monitoring import run_drift_monitoring
from src.train import (
    create_features,
    load_data_from_db,
    train_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
        time.sleep(5)
        logging.info("Docker Compose services started.")
    except Exception as e:
        logging.warning(
            f"Could not trigger Docker automatically: {e}. Ensure Docker Desktop is running."
        )


def verify_api_serving() -> None:
    """Tests the API /predict endpoint to confirm the deployed model works in runtime."""
    logging.info(
        "Step 7: Verifying REST API serving inference (/predict)..."
    )

    url = "http://localhost:8000/predict"
    payload = [
        {
            "store_id": 101,
            "product_id": 2045,
            "month": 10,
            "dayofweek": 2,
            "is_weekend": 0,
            "stock_on_hand": 500.0,
            "supplier_lead_time": 3.0,
            "lag_7": 120.5,
            "lag_14": 115.0,
            "lag_30": 110.0,
            "rolling_mean_7": 118.2,
            "rolling_mean_30": 112.4,
        }
    ]

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
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
                predicted_demand = result[0].get(
                    "predicted_demand"
                )
                logging.info(
                    f"✅ API Response Received: Predicted Demand = {predicted_demand:.2f}"
                )
            else:
                logging.error(
                    f"❌ API return status code: {response.status}"
                )
    except Exception as e:
        logging.error(f"❌ API Serving test failed: {e}")


def main() -> None:
    """Executes the full end-to-end MLOps pipeline."""
    logging.info(
        "=========================================="
    )
    logging.info("STARTING PHARMASUPPLY MLOPS PIPELINE")
    logging.info(
        "=========================================="
    )

    # 0. Infrastructure Check
    ensure_docker_running()

    # 1. Ingestion Automatique
    logging.info(
        "Step 1: Ingesting raw data into PostgreSQL..."
    )
    raw_data_path = Path(
        "data/pharmaceutical_demand_row.csv"
    )

    if not raw_data_path.exists():
        logging.info(
            "Synthetic dataset not found. Generating data..."
        )
        from generate_data import build_pharma_dataset

        df_pharma = build_pharma_dataset()
        raw_data_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        df_pharma.to_csv(raw_data_path, index=False)

    ingest_csv_to_postgres(
        file_path=raw_data_path,
        table_name="raw_pharmaceutical_demand",
    )

    # 2. Feature Engineering
    logging.info(
        "Step 2: Extracting data from DB & engineering features..."
    )
    df_raw = load_data_from_db()
    df_processed = create_features(df_raw)
    logging.info(
        f"Dataset successfully prepared. Total rows: {len(df_processed)}"
    )

    # 3. Model Training
    logging.info("Step 3: Training XGBoost model...")
    model, metrics, X_test, y_test = train_xgboost(
        df_processed
    )
    logging.info(
        f"Model trained successfully. Test MAE: {metrics['mae']:.4f}"
    )

    # 4. Explainability (SHAP)
    logging.info(
        "Step 4: Generating SHAP feature importance report..."
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

    # 5. Drift Monitoring
    logging.info(
        "Step 5: Running Evidently AI drift monitoring..."
    )
    run_drift_monitoring(
        df=df_processed,
        output_html_path="reports/drift_report.html",
    )

    # 6. REST API Infeference Validation
    verify_api_serving()

    logging.info(
        "=========================================="
    )
    logging.info("✅ PIPELINE EXECUTED SUCCESSFULLY!")
    logging.info("Generated Artifacts:")
    logging.info(
        " - Model artifact  : models/xgboost_pharma_demand.joblib"
    )
    logging.info(" - SHAP figures    : reports/figures/")
    logging.info(
        " - Drift report    : reports/drift_report.html"
    )
    logging.info(
        " - API Endpoint    : http://localhost:8000/predict"
    )
    logging.info(
        "=========================================="
    )


if __name__ == "__main__":
    main()
