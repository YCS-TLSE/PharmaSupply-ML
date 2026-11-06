"""
===============================================================================
PharmaSupply-ML: Data Ingestion & Staging Pipeline
===============================================================================

PURPOSE
-------
Load a raw pharmaceutical demand CSV into PostgreSQL, normalising optional
columns and preserving idempotence through an incremental UPSERT.

RESPONSIBILITIES
----------------
* Optionally ensure the Docker/PostgreSQL infrastructure is running.
* Read and normalise store/product identifiers and logistical columns.
* Align Python types with the PostgreSQL DDL (dates, integers, numerics).
* Create the natural-key unique index if an older schema lacks it.
* Upsert rows on ``(store_id, product_id, date)`` with ``execute_values``.

INPUTS / OUTPUTS
----------------
Inputs: a CSV path and a target table name.
Outputs: rows inserted/updated in PostgreSQL and human-readable progress logs.

ARCHITECTURE INTEGRATION
------------------------
Called by ``run_pipeline.py`` and the module CLI. The natural key matches
Alembic migration ``0001`` and ``sql/schema.sql``. Upsert semantics preserve
historical rows while refreshing changed values.
"""

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import subprocess
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MOQ,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SHELF_LIFE_DAYS,
)
from src.db_config import ROOT_DIR, get_database_url


# Infrastructure convenience: best-effort compose startup for local runs.
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


# SQLAlchemy engine used for schema inspection and the raw DBAPI connection
# required by execute_values.
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
# Security guard: the table name is interpolated into DDL/DML strings, so it
# must contain only letters, digits and underscores.
def _assert_safe_table_name(table_name: str) -> None:
    """Reject identifiers that cannot be interpolated into TRUNCATE safely."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError(
            f"Unsafe table name '{table_name}': use letters, digits, and underscores only."
        )


# Main ingestion routine: normalise, type-align, ensure the natural key, then
# perform a chunked UPSERT. Failure modes are reported without masking.
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
        df["safety_threshold"] = DEFAULT_SAFETY_THRESHOLD
    else:
        df["safety_threshold"] = df[
            "safety_threshold"
        ].astype(float)

    # --- 4. LOGISTICAL CONSTRAINTS (BATCH SIZE & MOQ) STAGING ---
    if "batch_size" not in df.columns:
        print(
            "ℹ️ Column 'batch_size' missing in source data: applying standard default (1 unit)"
        )
        df["batch_size"] = DEFAULT_BATCH_SIZE

    if "moq" not in df.columns:
        print(
            "ℹ️ Column 'moq' missing in source data: applying standard default (1 unit)"
        )
        df["moq"] = DEFAULT_MOQ

    # --- 5. EXPIRY CONSTRAINT (SHELF-LIFE) STAGING ---
    if "shelf_life_days" not in df.columns:
        print(
            "ℹ️ Column 'shelf_life_days' missing in source data: applying standard default (365 days)"
        )
        df["shelf_life_days"] = DEFAULT_SHELF_LIFE_DAYS
    else:
        df["shelf_life_days"] = (
            df["shelf_life_days"]
            .fillna(DEFAULT_SHELF_LIFE_DAYS)
            .astype(int)
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

    # --- 7. RELATIONAL DATABASE PERSISTENCE (INCREMENTAL UPSERT) ---
    try:
        _assert_safe_table_name(table_name)
        engine = get_db_engine()
        inspector = inspect(engine)

        if not inspector.has_table(table_name):
            print(
                f"❌ Error: Table '{table_name}' does not exist. "
                "Apply sql/schema.sql or `alembic upgrade head` before ingesting."
            )
            return

        table_columns = {
            column["name"]
            for column in inspector.get_columns(table_name)
        }
        # Let PostgreSQL populate SERIAL id and created_at defaults.
        insert_columns = [
            col
            for col in df.columns
            if col in table_columns
            and col not in {"id", "created_at"}
        ]
        if not insert_columns:
            print(
                f"❌ Error: No CSV columns match the schema of '{table_name}'."
            )
            return

        key_columns = {"store_id", "product_id", "date"}
        missing_keys = key_columns - set(insert_columns)
        if missing_keys:
            print(
                "❌ Error: Missing key columns required for UPSERT: "
                f"{sorted(missing_keys)}"
            )
            return

        df_to_load = df[insert_columns].copy()
        # Align Python types with the PostgreSQL DDL before binding.
        if "date" in df_to_load.columns:
            df_to_load["date"] = pd.to_datetime(
                df_to_load["date"]
            ).dt.date
        integer_columns = (
            "target_demand",
            "sales_volume",
            "stock_on_hand",
            "is_stockout",
            "supplier_lead_time",
            "batch_size",
            "moq",
            "shelf_life_days",
            "critical_expiry_days",
        )
        for column in integer_columns:
            if column in df_to_load.columns:
                df_to_load[column] = (
                    df_to_load[column].fillna(0).astype(int)
                )
        if "safety_threshold" in df_to_load.columns:
            df_to_load["safety_threshold"] = (
                df_to_load["safety_threshold"]
                .fillna(0)
                .astype(float)
            )

        print(
            f"⏳ Upserting {len(df_to_load)} records into "
            f"'{table_name}' (incremental)..."
        )

        # Ensure the natural key exists even on databases created from an
        # older schema.sql (Alembic migration 0001 also declares it).
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_pharma_store_product_date "
                    f"ON {table_name} (store_id, product_id, date);"
                )
            )

        columns_sql = ", ".join(insert_columns)
        update_columns = [
            column
            for column in insert_columns
            if column not in key_columns
        ]
        update_sql = ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in update_columns
        )
        conflict_target = "(store_id, product_id, date)"
        if update_sql:
            statement = (
                f"INSERT INTO {table_name} ({columns_sql}) VALUES %s "
                f"ON CONFLICT {conflict_target} DO UPDATE SET {update_sql}"
            )
        else:
            statement = (
                f"INSERT INTO {table_name} ({columns_sql}) VALUES %s "
                f"ON CONFLICT {conflict_target} DO NOTHING"
            )

        records = list(
            df_to_load.itertuples(index=False, name=None)
        )
        raw_connection = engine.raw_connection()
        try:
            with raw_connection.cursor() as cursor:
                execute_values(
                    cursor,
                    statement,
                    records,
                    page_size=1000,
                )
            raw_connection.commit()
        finally:
            raw_connection.close()

        print(
            f"✅ Success! {len(df_to_load)} rows upserted into "
            f"'{table_name}'."
        )
        print(
            f"📊 Active stores detected: {df['store_id'].unique().tolist()}"
        )
        print(
            f"💊 Active product catalogues detected: {df['product_id'].unique().tolist()}"
        )

    except (SQLAlchemyError, psycopg2.Error) as error:
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
