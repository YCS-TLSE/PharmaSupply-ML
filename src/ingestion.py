"""
===============================================================================
PharmaSupply-ML: Data Ingestion & Staging Pipeline
===============================================================================

OBJECTIVE & USAGE:
------------------
This script automates the staging and ingestion pipeline for the pharmaceutical
supply chain forecasting system. It provides:
1. Container Lifecycle Verification: Ensures the PostgreSQL Docker container is
   active and healthy prior to database operations.
2. Robust Data Normalization: Reads raw CSV historical datasets and standardizes
   identifiers (store IDs, product IDs) to prevent type mismatch errors.
3. Dynamic Constraint Integration: Detects and propagates user-provided custom
   safety thresholds, batch sizes, minimum order quantities (MOQ), shelf-life
   parameters, and critical expiry tracking (FEFO) per product, falling back
   gracefully to sensible default values.
4. Relational Database Loading: Persists the structured staging dataframe into
   PostgreSQL using SQLAlchemy for downstream feature engineering and model training.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import os
import subprocess
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Resolve project root directory dynamically to ensure environment-agnostic paths
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables explicitly from root .env configuration
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Retrieve database connection credentials matching environment specifications
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB")


def ensure_postgres_running():
    """
    Ensures that the PostgreSQL Docker container is up and running.
    Triggers 'docker compose up -d' as a preventative safeguard before connection attempts.
    """
    try:
        print(
            "🐳 Verifying PostgreSQL Docker container status..."
        )
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            check=True,
        )
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
    ) as error:
        print(
            f"⚠️ Warning: Could not verify Docker container status: {error}"
        )


def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for secure PostgreSQL connectivity.

    Returns:
        sqlalchemy.engine.Engine: Configured database connection engine.
    """
    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(database_url)


# ==============================================================================
# 2. INGESTION FUNCTION
# ==============================================================================
def ingest_csv_to_postgres(
    file_path: Path,
    table_name: str,
    if_exists: str = "replace",
):
    """
    Reads a raw CSV file via Pandas, normalizes operational features, and ingests
    its content into PostgreSQL using SQLAlchemy. Handles multi-store structures,
    custom safety thresholds, logistical constraints, and critical expiry tracking.
    """
    if not file_path.exists():
        print(f"❌ Error: File '{file_path}' not found.")
        return

    print(
        f"📦 Loading source dataset file: {file_path.name}"
    )
    df = pd.read_csv(file_path)

    # --- 1. STORE IDENTIFIER NORMALIZATION ---
    if "store_id" not in df.columns:
        print(
            "ℹ️ Column 'store_id' missing: assigning default store ('1')"
        )
        df["store_id"] = "1"
    else:
        df["store_id"] = df["store_id"].astype(str)

    # --- 2. PRODUCT CATALOGUE NORMALIZATION ---
    if "product_id" not in df.columns:
        print(
            "ℹ️ Column 'product_id' missing: assigning default product ('MED_001')"
        )
        df["product_id"] = "MED_001"
    else:
        df["product_id"] = df["product_id"].astype(str)

    # --- 3. SAFETY THRESHOLD PROPAGATION & CUSTOM FALLBACK ---
    if "safety_threshold" not in df.columns:
        print(
            "ℹ️ Column 'safety_threshold' missing in source data: applying standard default (150.0)"
        )
        df["safety_threshold"] = 150.0
    else:
        df["safety_threshold"] = df[
            "safety_threshold"
        ].astype(float)

    # --- 4. LOGISTICAL CONSTRAINTS (BATCH SIZE & MOQ) STAGING ---
    if "batch_size" not in df.columns:
        print(
            "ℹ️ Column 'batch_size' missing in source data: applying standard default (1 unit)"
        )
        df["batch_size"] = 1

    if "moq" not in df.columns:
        print(
            "ℹ️ Column 'moq' missing in source data: applying standard default (1 unit)"
        )
        df["moq"] = 1

    # --- 5. EXPIRY CONSTRAINT (SHELF-LIFE) STAGING ---
    if "shelf_life_days" not in df.columns:
        print(
            "ℹ️ Column 'shelf_life_days' missing in source data: applying standard default (365 days)"
        )
        df["shelf_life_days"] = 365
    else:
        df["shelf_life_days"] = (
            df["shelf_life_days"].fillna(365).astype(int)
        )

    # --- 6. CRITICAL EXPIRY CONSTRAINT (FEFO) STAGING ---
    if "critical_expiry_days" not in df.columns:
        print(
            "ℹ️ Column 'critical_expiry_days' missing in source data: defaulting to match 'shelf_life_days'"
        )
        df["critical_expiry_days"] = df["shelf_life_days"]
    else:
        df["critical_expiry_days"] = (
            df["critical_expiry_days"]
            .fillna(df["shelf_life_days"])
            .astype(int)
        )

    # --- 7. RELATIONAL DATABASE PERSISTENCE ---
    try:
        engine = get_db_engine()
        print(
            f"⏳ Ingesting {len(df)} structured records into staging table '{table_name}'..."
        )

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
        )
        print(
            f"✅ Success! {len(df)} rows successfully loaded into table '{table_name}'."
        )
        print(
            f"📊 Active stores detected: {df['store_id'].unique().tolist()}"
        )
        print(
            f"💊 Active product catalogues detected: {df['product_id'].unique().tolist()}"
        )

    except SQLAlchemyError as error:
        print(f"❌ Database ingestion error: {error}")


# ==============================================================================
# 3. EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Guarantee that database container infrastructure is active
    ensure_postgres_running()

    # Define path to raw staging CSV file inside root data/ directory
    raw_data_path = (
        ROOT_DIR / "data" / "pharmaceutical_demand_row.csv"
    )

    # Execute end-to-end ingestion pipeline into PostgreSQL staging table
    ingest_csv_to_postgres(
        file_path=raw_data_path,
        table_name="raw_pharmaceutical_demand",
    )
