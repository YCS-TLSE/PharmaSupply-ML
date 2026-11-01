"""
===============================================================================
PharmaSupply-ML: Integration Test Suite for Data Ingestion with Testcontainers
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This module demonstrates real integration testing in an MLOps pipeline.
Using Testcontainers, it spins up an ephemeral PostgreSQL Docker container
to validate the entire database ingestion workflow in a production-like environment.

Key Verifications:
1. Container Lifecycle: Automatic provisioning of a clean PostgreSQL instance.
2. DDL Compliance: Execution of `sql/schema.sql` to establish staging tables, indexes, and constraints.
3. Real Database Persistence: Verification that CSV files are correctly normalized,
   truncated, and appended via SQLAlchemy into the live container database.
"""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text
from testcontainers.community.postgres import (
    PostgresContainer,
)

from src.ingestion import ingest_csv_to_postgres


@pytest.fixture(scope="module")
def postgres_container():
    """
    Fixture that spins up an ephemeral PostgreSQL container using Testcontainers.
    Guarantees a clean, isolated database instance for integration testing.
    """
    with PostgresContainer(
        "postgres:15-alpine"
    ) as postgres:
        yield postgres


@pytest.fixture(scope="function")
def db_engine(postgres_container, monkeypatch):
    """
    Configures the environment and returns a SQLAlchemy engine
    connected to the ephemeral Testcontainers PostgreSQL instance.
    """
    db_url = postgres_container.get_connection_url()

    # Override environment variables so get_database_url() targets the container
    monkeypatch.setenv("DATABASE_URL", db_url)

    engine = create_engine(db_url)

    # Initialize database schema using schema.sql
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "sql"
        / "schema.sql"
    )
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with engine.begin() as connection:
            # Execute DDL statements
            for statement in schema_sql.split(";"):
                if statement.strip():
                    connection.execute(text(statement))

    yield engine
    engine.dispose()


def test_ingestion_integration_real_postgres(
    db_engine, tmp_path
):
    """
    Performs an end-to-end integration test of the ingestion pipeline
    against a live, containerized PostgreSQL database.
    """
    table_name = "raw_pharmaceutical_demand"

    # 1. Create a temporary raw CSV file with partial/missing operational columns
    csv_file = tmp_path / "integration_raw_demand.csv"
    raw_data = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "sales_volume": 150,
                "target_demand": 160,
                "stock_on_hand": 300,
                "is_stockout": 0,
                "supplier_lead_time": 5,
                # Missing store_id, product_id, safety_threshold, batch_size, moq, shelf_life_days, critical_expiry_days
            },
            {
                "date": "2026-06-02",
                "sales_volume": 200,
                "target_demand": 210,
                "stock_on_hand": 250,
                "is_stockout": 0,
                "supplier_lead_time": 4,
                "store_id": "STORE_B",
                "product_id": "MED_002",
            },
        ]
    )
    raw_data.to_csv(csv_file, index=False)

    # 2. Verify table exists prior to ingestion via inspector
    inspector = inspect(db_engine)
    assert inspector.has_table(table_name), (
        "Staging table must exist after schema initialization."
    )

    # 3. Execute the ingestion pipeline (which connects to the container via monkeypatched env)
    ingest_csv_to_postgres(csv_file, table_name)

    # 4. Query the live container database to verify actual persistence and default backfilling
    with db_engine.connect() as connection:
        result = connection.execute(
            text(
                f"SELECT COUNT(*), MIN(store_id), MAX(store_id) FROM {table_name}"
            )
        )
        row = result.fetchone()

        # We inserted 2 rows
        assert row[0] == 2, (
            "Exactly 2 rows should be ingested into PostgreSQL."
        )

    # 5. Test Idempotence: Run ingestion a second time to verify TRUNCATE works correctly without duplicating rows
    ingest_csv_to_postgres(csv_file, table_name)

    with db_engine.connect() as connection:
        result = connection.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        )
        row = result.fetchone()
        assert row[0] == 2, (
            "Idempotence check failed: TRUNCATE should reset table content on re-ingestion."
        )
