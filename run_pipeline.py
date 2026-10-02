"""Master MLOps Pipeline Orchestrator.

Sequentially executes:
1. Docker infrastructure check/startup.
2. Raw data generation (if missing) & PostgreSQL ingestion.
3. Feature engineering & data extraction from DB.
4. Model training & evaluation.
5. SHAP Explainability report.
6. Evidently AI Drift Monitoring.
"""

import logging
import socket
import subprocess
import time
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


def ensure_docker_running() -> None:
    """Verifies PostgreSQL container availability via socket check and attempts auto-start if down."""
    logging.info(
        "Step 0: Checking PostgreSQL connectivity on port 5432..."
    )

    # 1. Test de connexion direct sur le port 5432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    is_open = sock.connect_ex(("127.0.0.1", 5432)) == 0
    sock.close()

    if is_open:
        logging.info(
            "PostgreSQL service is online and accepting connections."
        )
        return

    # 2. Tentative de lancement si le port n'est pas ouvert
    logging.info(
        "Port 5432 unreachable. Attempting to start Docker containers..."
    )
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"], check=True
        )
        # Laisser un court délai pour que PostgreSQL soit prêt à accepter des connexions
        time.sleep(5)
        logging.info("Docker Compose services started.")
    except Exception as e:
        logging.warning(
            f"Could not trigger Docker automatically: {e}. Ensure Docker Desktop and Postgres are running."
        )


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

    # 1. Ingestion Automatique (Génération et/ou Chargement en BDD)
    logging.info(
        "Step 1: Ingesting raw data into PostgreSQL..."
    )
    raw_data_path = Path(
        "data/pharmaceutical_demand_row.csv"
    )

    # Si le fichier CSV brut n'existe pas, on le génère à la volée
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

    # Ingestion dans PostgreSQL (crée la table et insère les données)
    ingest_csv_to_postgres(
        file_path=raw_data_path,
        table_name="raw_pharmaceutical_demand",
    )

    # 2. Feature Engineering & Chargement depuis la base
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
        " - Drift report     : reports/drift_report.html"
    )
    logging.info(
        "=========================================="
    )


if __name__ == "__main__":
    main()
