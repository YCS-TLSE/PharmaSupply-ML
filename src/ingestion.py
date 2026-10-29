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
4. Relational Database Loading: Truncates the existing staging table (preserving
   PostgreSQL DDL, SERIAL keys, and indexes) then appends rows via SQLAlchemy.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.db_config import ROOT_DIR, get_database_url


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
    return create_engine(get_database_url())


# ==============================================================================
# 2. INGESTION FUNCTION
# ==============================================================================
def _assert_safe_table_name(table_name: str) -> None:
    """Reject identifiers that cannot be interpolated into TRUNCATE safely."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError(
            f"Unsafe table name '{table_name}': use letters, digits, and underscores only."
        )


def ingest_csv_to_postgres(
    file_path: Path,
    table_name: str,
):
    """
    Reads a raw CSV file via Pandas, normalizes operational features, and ingests
    its content into PostgreSQL using SQLAlchemy. Truncates the target table
    first so PostgreSQL DDL (types, SERIAL PK, indexes) is preserved.
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

    # --- 7. RELATIONAL DATABASE PERSISTENCE (TRUNCATE + APPEND) ---
    try:
        _assert_safe_table_name(table_name)
        engine = get_db_engine()
        inspector = inspect(engine)

        if not inspector.has_table(table_name):
            print(
                f"❌ Error: Table '{table_name}' does not exist. "
                "Apply sql/schema.sql before ingesting."
            )
            return

        table_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        # Let PostgreSQL populate SERIAL id and created_at defaults.
        insert_columns = [
            col
            for col in df.columns
            if col in table_columns and col not in {"id", "created_at"}
        ]
        if not insert_columns:
            print(
                f"❌ Error: No CSV columns match the schema of '{table_name}'."
            )
            return

        df_to_load = df[insert_columns]
        print(
            f"⏳ Truncating '{table_name}' (DDL preserved) then appending "
            f"{len(df_to_load)} records..."
        )

        with engine.begin() as connection:
            connection.execute(
                text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
            )

        df_to_load.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False,
        )

        print(
            f"✅ Success! {len(df_to_load)} rows loaded into '{table_name}' "
            "without replacing the table definition."
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
