"""
Data ingestion module for PharmaSupply-ML.
Loads raw CSV datasets from data/ into PostgreSQL.
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

# Resolve project root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables explicitly from root .env
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Retrieve database credentials matching your .env file
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB")


def ensure_postgres_running():
    """
    Ensures that the PostgreSQL Docker container is up and running.
    Triggers the custom Docker wrapper via interactive Zsh shell.
    """
    try:
        print("🐳 Verifying PostgreSQL Docker container status...")
        # Force loading interactive shell aliases/functions via zsh -i
        subprocess.run(
            ["zsh", "-i", "-c", "docker compose up -d"],
            check=True,
        )
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
    ) as error:
        print(f"⚠️ Warning: Could not verify Docker container status: {error}")


def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for PostgreSQL connection.

    Returns:
        sqlalchemy.engine.Engine: Database connection engine.
    """
    # Assemble standard URI: postgresql://user:password@host:port/dbname
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
    Reads a CSV file via Pandas and ingests its content into PostgreSQL using SQLAlchemy.

    Args:
        file_path (Path): Absolute or relative path to the source CSV file.
        table_name (str): Target table name in the PostgreSQL database.
        if_exists (str): Behavior if table already exists ('replace', 'append', 'fail').
    """
    # Check if the source file exists before reading
    if not file_path.exists():
        print(f"❌ Error: File '{file_path}' not found.")
        return

    print(f"📦 Loading file: {file_path.name}")
    df = pd.read_csv(file_path)

    try:
        engine = get_db_engine()
        print(f"⏳ Ingesting {len(df)} rows into table '{table_name}'...")

        # Perform SQL ingestion (index=False prevents creating a Pandas index column in SQL)
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
        )
        print(f"✅ Success! {len(df)} rows inserted into table '{table_name}'.")

    except SQLAlchemyError as error:
        # Catch database-related issues (connection failures, schema errors)
        print(f"❌ Database error: {error}")


# ==============================================================================
# 3. EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Ensure database container is active before executing ingestion
    ensure_postgres_running()

    # Define path to raw CSV file inside root data/ directory
    raw_data_path = ROOT_DIR / "data" / "pharmaceutical_demand_row.csv"

    # Execute ingestion into staging table
    ingest_csv_to_postgres(
        file_path=raw_data_path,
        table_name="raw_pharmaceutical_demand",
    )
