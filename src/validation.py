"""
Data Validation Module for PharmaSupply-ML Pipeline.

This script connects to the PostgreSQL database and performs automated
quality checks on the ingested raw data to guarantee data integrity,
schema compliance, and completeness before feature engineering.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Build PostgreSQL connection URL from environment variables
DB_USER = os.getenv("POSTGRES_USER", "ycs_admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ycs_admin_secure_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "pharmasupply_db")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def run_validation():
    """
    Execute data quality checks on the 'raw_pharmaceutical_demand' table.

    Validations executed:
      1. Non-empty table check.
      2. Null value assertions on key schema columns.
      3. Logical range assertions (demand and stock must be non-negative).

    Raises:
        AssertionError: If any data quality check fails.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM raw_pharmaceutical_demand", engine)

    print(f"🔍 Validating {len(df)} rows...")

    # 1. Ensure table is not empty
    assert len(df) > 0, "Data table is empty!"

    # 2. Check for missing values in critical columns
    required_cols = [
        "date",
        "product_id",
        "target_demand",
        "sales_volume",
        "stock_on_hand",
    ]
    null_counts = df[required_cols].isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} null values!"

    # 3. Check for invalid negative values
    assert (df["target_demand"] >= 0).all(), "Found negative target_demand values!"
    assert (df["stock_on_hand"] >= 0).all(), "Found negative stock_on_hand values!"

    print("✅ All data quality checks passed successfully!")


if __name__ == "__main__":
    run_validation()
