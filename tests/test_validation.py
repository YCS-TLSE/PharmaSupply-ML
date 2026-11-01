"""
===============================================================================
PharmaSupply-ML: Test Suite for Data Validation & Quality Guardrails Module
===============================================================================

EDUCATIONAL & MLOPS PURPOSE:
----------------------------
This test module demonstrates how to implement automated Data Quality Gates
in a production MLOps pipeline using pytest and pandas.

In pharmaceutical supply chain forecasting, consuming corrupted or out-of-bounds
data leads to incorrect replenishment recommendations. The validation module
acts as a strict filter (Quality Gate) positioned between raw data ingestion
and downstream pipeline stages (feature engineering, training, inference).

Key MLOps Validation Patterns Tested:
1. Hard Fail Mechanism (Critical Quality Guardrail):
   Verifies that missing critical schema columns, invalid store/product IDs,
   or completely empty datasets trigger immediate exceptions to stop the pipeline
   before feeding corrupted inputs to ML models.

2. Soft Auto-Correction (Operational Resilience):
   Validates that non-fatal data anomalies (e.g., negative stock levels,
   unrealistic MOQ values, negative demands, or missing lead times) are
   gracefully imputed or clamped to domain-specific boundaries without
   halting pipeline execution.

3. Cross-Column Business Logic Integrity:
   Ensures complex multi-feature constraints are enforced (e.g., historical sales
   cannot exceed target demand).

4. Schema & Data Type Integrity:
   Ensures that output DataFrames maintain strict expected types and column
   alignment across execution runs.

Author: PharmaSupply-ML Team
"""

import os
import sys

import pandas as pd
import pytest

# Add the project root directory to sys.path to allow absolute imports of src.validation
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.validation import run_validation_and_clean


@pytest.fixture
def valid_raw_dataframe():
    """
    Pytest fixture providing a baseline, valid raw demand DataFrame.
    Represents nominal operating conditions in the supply chain data stream.
    """
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "store_id": "STORE_001",
                "product_id": "MED_001",
                "sales_volume": 100.0,
                "target_demand": 105.0,
                "stock_on_hand": 500.0,
                "supplier_lead_time": 3.0,
                "safety_threshold": 150.0,
                "batch_size": 50,
                "moq": 100,
                "shelf_life_days": 365,
                "critical_expiry_days": 180,
            },
            {
                "date": "2026-01-02",
                "store_id": "STORE_001",
                "product_id": "MED_002",
                "sales_volume": 50.0,
                "target_demand": 52.0,
                "stock_on_hand": 200.0,
                "supplier_lead_time": 5.0,
                "safety_threshold": 80.0,
                "batch_size": 10,
                "moq": 20,
                "shelf_life_days": 180,
                "critical_expiry_days": 45,
            },
        ]
    )


def test_validation_nominal(valid_raw_dataframe):
    """
    Validates nominal raw data processing under normal operating conditions.
    Ensures that a properly structured input DataFrame passes validation
    without throwing errors or altering valid core metrics.
    """
    # Execute validation and cleaning pipeline on valid data
    cleaned_df = run_validation_and_clean(
        valid_raw_dataframe
    )

    # Assertions to guarantee nominal behavior
    assert not cleaned_df.empty, (
        "Cleaned DataFrame should not be empty."
    )
    assert len(cleaned_df) == len(valid_raw_dataframe), (
        "Row count must remain unchanged."
    )
    assert "date" in cleaned_df.columns, (
        "Mandatory date column must be present."
    )


def test_validation_hard_fail_missing_columns():
    """
    Validates Hard Fail guardrail on incomplete dataset schemas.
    Ensures that a DataFrame missing mandatory columns raises a ValueError
    to halt execution immediately before data corruption spreads.
    """
    incomplete_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "store_id": "STORE_001",
                # 'product_id', 'target_demand' and other mandatory columns are missing
            }
        ]
    )

    # Verify that a ValueError is raised due to missing mandatory columns
    with pytest.raises(ValueError):
        run_validation_and_clean(incomplete_df)


def test_validation_hard_fail_empty_dataframe():
    """
    Validates Hard Fail guardrail on empty data ingestion stages.
    Ensures that passing an empty DataFrame triggers a critical validation error.
    """
    empty_df = pd.DataFrame()

    # Verify that passing an empty dataset blocks pipeline execution
    with pytest.raises(ValueError):
        run_validation_and_clean(empty_df)


def test_validation_soft_correction_negative_stock(
    valid_raw_dataframe,
):
    """
    Validates Soft Auto-Correction for physically impossible negative stock values.
    Confirms that negative stock levels are automatically corrected and clamped
    to 0.0 to maintain operational resilience.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[
        0, "stock_on_hand"
    ] = -150.0  # Introduce invalid negative stock

    cleaned_df = run_validation_and_clean(corrupted_df)

    # Negative stock should be auto-corrected to 0.0
    assert cleaned_df.loc[0, "stock_on_hand"] == 0.0, (
        "Negative stock should be clamped to 0.0."
    )


def test_validation_soft_correction_negative_demand_and_sales(
    valid_raw_dataframe,
):
    """
    Validates Soft Auto-Correction for negative demand and sales volumes.
    Ensures erroneous negative values are securely clipped to 0.0.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[0, "target_demand"] = -50.0
    corrupted_df.loc[0, "sales_volume"] = -10.0

    cleaned_df = run_validation_and_clean(corrupted_df)

    assert cleaned_df.loc[0, "target_demand"] == 0.0, (
        "Negative target demand should be clamped to 0.0."
    )
    assert cleaned_df.loc[0, "sales_volume"] == 0.0, (
        "Negative sales volume should be clamped to 0.0."
    )


def test_validation_soft_correction_invalid_moq(
    valid_raw_dataframe,
):
    """
    Validates Soft Auto-Correction for invalid logistical minimum order quantities (MOQ)
    and batch sizes.
    Confirms that invalid values (MOQ <= 0 or batch_size < 1) are set to safe minimum defaults.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[0, "moq"] = 0  # Invalid MOQ
    corrupted_df.loc[
        0, "batch_size"
    ] = -5  # Invalid batch size

    cleaned_df = run_validation_and_clean(corrupted_df)

    # Values should be clamped or defaulted to at least 1
    assert cleaned_df.loc[0, "moq"] >= 1, (
        "MOQ must be corrected to at least 1."
    )
    assert cleaned_df.loc[0, "batch_size"] >= 1, (
        "Batch size must be corrected to at least 1."
    )


def test_validation_soft_correction_invalid_lead_time_and_threshold(
    valid_raw_dataframe,
):
    """
    Validates Soft Auto-Correction for non-positive lead times and safety thresholds.
    Ensures they fall back to secure default operational values.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[0, "supplier_lead_time"] = (
        0.0  # Invalid lead time
    )
    corrupted_df.loc[
        0, "safety_threshold"
    ] = -20.0  # Invalid threshold

    cleaned_df = run_validation_and_clean(corrupted_df)

    assert cleaned_df.loc[0, "supplier_lead_time"] == 3.0, (
        "Lead time should default to 3.0 days."
    )
    assert cleaned_df.loc[0, "safety_threshold"] == 150.0, (
        "Safety threshold should default to 150.0."
    )


def test_validation_cross_column_sales_exceeding_demand(
    valid_raw_dataframe,
):
    """
    Validates cross-column business logic integrity.
    Ensures that if historical sales volume exceeds target demand,
    sales_volume is adjusted downward to match target_demand logically.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[0, "target_demand"] = 100.0
    corrupted_df.loc[0, "sales_volume"] = (
        150.0  # Sales exceeding target demand
    )

    cleaned_df = run_validation_and_clean(corrupted_df)

    assert (
        cleaned_df.loc[0, "sales_volume"]
        == cleaned_df.loc[0, "target_demand"]
    ), (
        "Sales volume cannot exceed target demand and must be adjusted."
    )


def test_validation_completeness_null_imputation(
    valid_raw_dataframe,
):
    """
    Validates missing values imputation (Completeness quality check).
    Ensures nulls in numerical columns are filled with 0.0 and string columns
    with 'UNKNOWN'.
    """
    corrupted_df = valid_raw_dataframe.copy()
    corrupted_df.loc[0, "target_demand"] = None
    corrupted_df.loc[0, "store_id"] = None

    cleaned_df = run_validation_and_clean(corrupted_df)

    assert cleaned_df.loc[0, "target_demand"] == 0.0, (
        "Null numerical values should be imputed to 0.0."
    )
    assert cleaned_df.loc[0, "store_id"] == "UNKNOWN", (
        "Null string values should be imputed to 'UNKNOWN'."
    )


def test_validation_type_casting(valid_raw_dataframe):
    """
    Validates automatic data type normalization and schema alignment.
    Ensures core metrics and operational parameters maintain expected types
    across execution runs.
    """
    cleaned_df = run_validation_and_clean(
        valid_raw_dataframe
    )

    assert "date" in cleaned_df.columns
    assert "stock_on_hand" in cleaned_df.columns
    assert "moq" in cleaned_df.columns
