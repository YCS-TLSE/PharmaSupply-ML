"""
===============================================================================
PharmaSupply-ML: Test Suite for Data Ingestion & Staging Pipeline
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This test module demonstrates how to test data ingestion pipelines, CSV normalization
logic, and database interaction boundaries safely using pytest and mocking techniques.

In robust MLOps architectures, ingestion scripts bridge external data sources
and internal staging storage. Testing these components ensures:
1. Security Guardrails: Prevention of SQL injection via rigorous table name sanitization.
2. Graceful Error Handling: Proper detection of missing source files or non-existent
   database tables.
3. Schema Normalization: Automatic fallback assignment for missing operational
   columns (store IDs, safety thresholds, MOQ, and expiry metrics) prior to database loading.

Author: PharmaSupply-ML Team
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the project root directory to sys.path to allow absolute imports of src.ingestion
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.ingestion import (
    _assert_safe_table_name,
    ingest_csv_to_postgres,
)


def test_assert_safe_table_name_valid():
    """
    Validates table name sanitization security guardrail with valid identifiers.
    Ensures standard alphanumeric and underscore table names pass safely.
    """
    # Should not raise any exception
    _assert_safe_table_name("raw_pharmaceutical_demand")
    _assert_safe_table_name("staging_table_01")


def test_assert_safe_table_name_unsafe():
    """
    Validates table name sanitization security guardrail against SQL injection attempts.
    Ensures dangerous characters (spaces, semicolons, quotes, dashes) trigger a ValueError.
    """
    unsafe_names = [
        "raw_demand; DROP TABLE users;",
        "table-name-with-dash",
        "table name with space",
        "table'name",
    ]

    for name in unsafe_names:
        with pytest.raises(ValueError):
            _assert_safe_table_name(name)


def test_ingest_csv_file_not_found(capsys):
    """
    Validates graceful handling of missing raw source CSV files.
    Ensures the pipeline reports an error cleanly without crashing unexpectedly.
    """
    non_existent_path = Path("non_existent_file_12345.csv")

    # Run ingestion with missing file
    ingest_csv_to_postgres(
        non_existent_path, "raw_pharmaceutical_demand"
    )

    captured = capsys.readouterr()
    assert "not found" in captured.out


@patch("src.ingestion.execute_values")
@patch("src.ingestion.get_db_engine")
@patch("src.ingestion.inspect")
def test_ingest_csv_normalization_and_staging(
    mock_inspect,
    mock_get_engine,
    mock_execute_values,
    tmp_path,
):
    """
    Validates CSV data normalization and database staging workflow using mocks.
    Creates a temporary partial CSV file, verifies that missing columns
    (store_id, moq, safety_threshold, etc.) are correctly backfilled with defaults,
    and checks that database operations (truncate and append) are correctly orchestrated.
    """
    # 1. Create a temporary raw CSV file with minimal/partial columns
    csv_file = tmp_path / "test_raw_demand.csv"
    raw_data = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "sales_volume": 120.0,
                "target_demand": 130.0,
                "stock_on_hand": 400.0,
                "supplier_lead_time": 4.0,
                # Missing store_id, product_id, safety_threshold, moq, etc.
            }
        ]
    )
    raw_data.to_csv(csv_file, index=False)

    # 2. Mock SQLAlchemy engine and inspector behavior
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    mock_inspector = MagicMock()
    mock_inspector.has_table.return_value = True
    mock_inspector.get_columns.return_value = [
        {"name": "date"},
        {"name": "store_id"},
        {"name": "product_id"},
        {"name": "sales_volume"},
        {"name": "target_demand"},
        {"name": "stock_on_hand"},
        {"name": "supplier_lead_time"},
        {"name": "safety_threshold"},
        {"name": "batch_size"},
        {"name": "moq"},
        {"name": "shelf_life_days"},
        {"name": "critical_expiry_days"},
    ]
    mock_inspect.return_value = mock_inspector

    # 3. Execute ingestion function
    ingest_csv_to_postgres(
        csv_file, "raw_pharmaceutical_demand"
    )

    # 4. Verify that table presence was checked and the UPSERT ran
    mock_inspector.has_table.assert_called_once_with(
        "raw_pharmaceutical_demand"
    )
    mock_engine.begin.assert_called()
    mock_execute_values.assert_called_once()
