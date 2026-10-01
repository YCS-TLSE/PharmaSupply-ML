"""Module for MLOps Data and Target Drift Monitoring using Evidently AI.

CONTEXT & PURPOSE:
------------------
Compares baseline (training) reference data against new incoming or current test data
to detect feature distribution drift and target drift.
"""

import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.train import create_features, load_data_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FEATURE_COLS = [
    "month",
    "dayofweek",
    "is_weekend",
    "stock_on_hand",
    "supplier_lead_time",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]
TARGET_COL = "target_demand"


def run_drift_monitoring(
    df: pd.DataFrame = None,
    output_html_path: str = "reports/drift_report.html",
) -> None:
    """Calculates data and target drift using Evidently AI and saves an HTML report."""
    if df is None:
        logging.info(
            "Loading dataset for monitoring from PostgreSQL..."
        )
        df_raw = load_data_from_db()
        df = create_features(df_raw)

    unique_dates = df["date"].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    reference_df = df[df["date"] < split_date][
        FEATURE_COLS + [TARGET_COL]
    ]
    current_df = df[df["date"] >= split_date][
        FEATURE_COLS + [TARGET_COL]
    ]

    logging.info(
        f"Reference set size: {len(reference_df)} rows | Current set size: {len(current_df)} rows"
    )

    logging.info("Generating Evidently AI Drift Report...")
    # syntaxe officielle Evidently v0.6+
    report = Report([DataDriftPreset()])
    my_eval = report.run(current_df, reference_df)

    report_path = Path(output_html_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde sur l'objet retourné par report.run()
    my_eval.save_html(str(report_path))

    logging.info(
        f"✅ Monitoring report successfully saved -> {report_path}"
    )


if __name__ == "__main__":
    run_drift_monitoring()
