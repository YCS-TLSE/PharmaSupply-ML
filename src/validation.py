"""
===============================================================================
PharmaSupply-ML: Autonomous Data Validation & Auto-Correction
===============================================================================

PURPOSE
-------
Act as a quality gate between raw ingestion and feature engineering. The gate
distinguishes structural corruption (blocking) from repairable anomalies
(non-blocking).

RESPONSIBILITIES
----------------
* Hard fail on empty datasets and missing mandatory columns.
* Impute missing required values with typed defaults.
* Clamp impossible values (negative stock, demand, sales, lead times).
* Enforce logistical bounds (batch size, MOQ, shelf life, expiry).
* Enforce cross-column business rules (sales cannot exceed target demand).
* Add default values for optional FEFO/logistics columns.

INPUTS / OUTPUTS
----------------
Inputs: a DataFrame from PostgreSQL or an in-memory fixture for tests.
Outputs: a cleaned DataFrame with stable types and documented defaults.

ARCHITECTURE INTEGRATION
------------------------
Used by ``src.train.load_validated_data`` and ``run_pipeline.py``. It is
intentionally independent from FastAPI so the same rules can be tested and
reused in batch jobs.
"""

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MOQ,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SHELF_LIFE_DAYS,
    DEFAULT_SUPPLIER_LEAD_TIME,
)
from src.db_config import get_database_url


# Entry point of the quality gate. ``input_df`` enables fast unit tests without
# a live database; when omitted, data is read from the staging table.
def run_validation_and_clean(
    input_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Executes autonomous data quality checks, applies auto-corrections for minor
    anomalies, and blocks execution only on critical structural failures.

    Accepts an optional input_df for unit testing purposes.
    """
    if input_df is not None:
        df = input_df.copy()
    else:
        engine = create_engine(get_database_url())
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT * FROM raw_pharmaceutical_demand"
                ),
                con=conn,
            )

    print(
        f"🔍 Running autonomous quality gate on {len(df)} records..."
    )

    # --- 1. CRITICAL STRUCTURAL CHECKS (BLOCKING) ---
    if len(df) == 0:
        raise ValueError(
            "❌ Critical Error: Data staging table is empty!"
        )

    required_cols = [
        "date",
        "store_id",
        "product_id",
        "target_demand",
        "sales_volume",
        "stock_on_hand",
        "supplier_lead_time",
        "safety_threshold",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"❌ Critical Error: Missing mandatory columns in dataset: {missing_cols}"
        )

    # --- 2. COMPLETENESS CHECK (MISSING VALUES IMPUTATION) ---
    for col in required_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(
                f"⚠️ Warning: Found {null_count} null values in '{col}'. Applying smart imputation..."
            )
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("UNKNOWN")

    # --- 3. OPTIONAL LOGISTICAL & EXPIRY COLUMNS CHECK & FALLBACK ---
    if "batch_size" not in df.columns:
        print(
            "ℹ️ Column 'batch_size' absent from table: defaulting to 1 unit."
        )
        df["batch_size"] = DEFAULT_BATCH_SIZE
    else:
        df["batch_size"] = (
            df["batch_size"]
            .fillna(DEFAULT_BATCH_SIZE)
            .astype(int)
        )

    if "moq" not in df.columns:
        print(
            "ℹ️ Column 'moq' absent from table: defaulting to 1 unit."
        )
        df["moq"] = DEFAULT_MOQ
    else:
        df["moq"] = (
            df["moq"].fillna(DEFAULT_MOQ).astype(int)
        )

    if "shelf_life_days" not in df.columns:
        print(
            "ℹ️ Column 'shelf_life_days' absent from table: defaulting to 365 days."
        )
        df["shelf_life_days"] = DEFAULT_SHELF_LIFE_DAYS
    else:
        df["shelf_life_days"] = (
            df["shelf_life_days"]
            .fillna(DEFAULT_SHELF_LIFE_DAYS)
            .astype(int)
        )

    # Fallback & validation for critical expiry tracking (FEFO preparation)
    if "critical_expiry_days" not in df.columns:
        print(
            "ℹ️ Column 'critical_expiry_days' absent from table: defaulting to match 'shelf_life_days'."
        )
        df["critical_expiry_days"] = df["shelf_life_days"]
    else:
        df["critical_expiry_days"] = (
            df["critical_expiry_days"]
            .fillna(df["shelf_life_days"])
            .astype(int)
        )

    # --- 4. SMART AUTO-CORRECTION FOR BUSINESS RULES (SOFT FIXES) ---

    # Fix negative target_demand
    neg_demand = (df["target_demand"] < 0).sum()
    if neg_demand > 0:
        print(
            f"🛠️ Auto-correction: Fixed {neg_demand} negative 'target_demand' values (set to 0)."
        )
        df.loc[df["target_demand"] < 0, "target_demand"] = 0

    # Fix negative sales_volume
    neg_sales = (df["sales_volume"] < 0).sum()
    if neg_sales > 0:
        print(
            f"🛠️ Auto-correction: Fixed {neg_sales} negative 'sales_volume' values (set to 0)."
        )
        df.loc[df["sales_volume"] < 0, "sales_volume"] = 0

    # Fix negative stock_on_hand
    neg_stock = (df["stock_on_hand"] < 0).sum()
    if neg_stock > 0:
        print(
            f"🛠️ Auto-correction: Fixed {neg_stock} negative 'stock_on_hand' values (set to 0)."
        )
        df.loc[df["stock_on_hand"] < 0, "stock_on_hand"] = 0

    # Fix invalid supplier_lead_time (must be positive, default to 3 if <= 0)
    invalid_lt = (df["supplier_lead_time"] <= 0).sum()
    if invalid_lt > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_lt} non-positive supplier lead times (set to default: 3 days)."
        )
        df.loc[
            df["supplier_lead_time"] <= 0,
            "supplier_lead_time",
        ] = DEFAULT_SUPPLIER_LEAD_TIME

    # Fix invalid safety_threshold (must be positive, default to 150.0 if <= 0)
    invalid_st = (df["safety_threshold"] <= 0).sum()
    if invalid_st > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_st} non-positive safety thresholds (set to default: 150.0)."
        )
        df.loc[
            df["safety_threshold"] <= 0, "safety_threshold"
        ] = DEFAULT_SAFETY_THRESHOLD

    # Fix invalid batch_size (must be >= 1)
    invalid_bs = (df["batch_size"] < 1).sum()
    if invalid_bs > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_bs} invalid batch sizes (set to default: 1)."
        )
        df.loc[df["batch_size"] < 1, "batch_size"] = (
            DEFAULT_BATCH_SIZE
        )

    # Fix invalid moq (must be >= 1)
    invalid_moq = (df["moq"] < 1).sum()
    if invalid_moq > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_moq} invalid MOQ values (set to default: 1)."
        )
        df.loc[df["moq"] < 1, "moq"] = DEFAULT_MOQ

    # Fix invalid shelf_life_days (must be >= 1)
    invalid_slf = (df["shelf_life_days"] < 1).sum()
    if invalid_slf > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_slf} invalid shelf-life values (set to default: 365)."
        )
        df.loc[
            df["shelf_life_days"] < 1, "shelf_life_days"
        ] = DEFAULT_SHELF_LIFE_DAYS

    # Fix invalid critical_expiry_days (must be >= 0)
    invalid_ced = (df["critical_expiry_days"] < 0).sum()
    if invalid_ced > 0:
        print(
            f"🛠️ Auto-correction: Fixed {invalid_ced} invalid critical expiry days (set to 0)."
        )
        df.loc[
            df["critical_expiry_days"] < 0,
            "critical_expiry_days",
        ] = 0

    # Cross-column logic fix: Sales cannot exceed true demand
    invalid_sales_mask = (
        df["sales_volume"] > df["target_demand"]
    )
    invalid_sales_count = invalid_sales_mask.sum()
    if invalid_sales_count > 0:
        print(
            f"🛠️ Auto-correction: Adjusted {invalid_sales_count} rows where 'sales_volume' exceeded 'target_demand'."
        )
        df.loc[invalid_sales_mask, "sales_volume"] = df.loc[
            invalid_sales_mask, "target_demand"
        ]

    print(
        "✅ Autonomous validation and auto-correction pipeline completed successfully!"
    )
    return df


if __name__ == "__main__":
    run_validation_and_clean()
