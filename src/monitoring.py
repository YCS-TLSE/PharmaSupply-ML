"""
===============================================================================
PharmaSupply-ML: MLOps Data Drift Monitoring Module
===============================================================================

OBJECTIVE & USAGE:
------------------
This script automates the production drift evaluation pipeline for the demand
forecasting model. Its sole responsibility is statistical analysis:
1. Data Ingestion & Splitting: Fetches historical records from PostgreSQL and splits
   them chronologically into reference (baseline) and current production streams.
2. Statistical Drift Evaluation: Generates an Evidently AI DataDriftPreset report
   and extracts column-level drift metrics.
3. Weighted Criticality Scoring: Multiplies feature-level drift flags by domain/SHAP
   weights (e.g., lag_7 and stock_on_hand carry higher weights than calendar attributes).
4. State Persistence: Writes the evaluated raw status string ('STABLE', 'WARNING', or
   'CRITICAL') into a shared persistence file ('model_status.txt').

Downstream notifications and user messaging are handled independently by the
FastAPI serving layer upon consuming this file.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import logging
import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.train import create_features, load_validated_data

# Configure standard logging format for automated batch executions
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Feature columns tracked by the forecasting model
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

# Path configuration for shared model status persistence across services
ROOT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_STATUS_FILE = ROOT_DIR / "model_status.txt"
STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH",
    str(_DEFAULT_STATUS_FILE) if _DEFAULT_STATUS_FILE.exists() else "model_status.txt",
)

# Threshold definitions for the weighted drift score
WARNING_THRESHOLD = (
    0.15  # 15% weighted drift -> WARNING status
)
CRITICAL_THRESHOLD = (
    0.30  # 30% weighted drift -> CRITICAL status
)


def run_drift_monitoring(
    df: pd.DataFrame = None,
    output_html_path: str = "reports/drift_report.html",
) -> None:
    """
    Executes the data drift evaluation pipeline, computes the weighted criticality
    score, generates the HTML report, and updates the shared status file.

    Parameters:
        df (pd.DataFrame, optional): Pre-loaded feature dataframe. If None, fetches from DB.
        output_html_path (str): File path where the Evidently HTML report will be saved.
    """
    # 1. Data Ingestion & Chronological Split
    if df is None:
        logging.info(
            "Fetching raw dataset from PostgreSQL database..."
        )
        df_raw = load_validated_data()
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

    # 2. Evidently AI Report Generation
    logging.info(
        "Generating Evidently AI comprehensive drift report..."
    )
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(
        reference_data=reference_df, current_data=current_df
    )

    report_path = Path(output_html_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    target_obj = my_eval if my_eval is not None else report
    if hasattr(target_obj, "save_html"):
        target_obj.save_html(str(report_path))
    elif hasattr(target_obj, "save"):
        target_obj.save(str(report_path))
    elif hasattr(target_obj, "get_html"):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(target_obj.get_html())

    # 3. Feature-Level Drift Extraction
    try:
        eval_obj = (
            my_eval if my_eval is not None else report
        )
        result_dict = (
            eval_obj.as_dict()
            if hasattr(eval_obj, "as_dict")
            else eval_obj.dict()
        )

        metrics_list = result_dict.get("metrics", [])
        drift_by_column = {}
        for m in metrics_list:
            if "DatasetDriftMetric" in m.get("metric", ""):
                drift_by_column = m.get("result", {}).get(
                    "drift_by_columns", {}
                )
                break
    except Exception as err:
        logging.warning(
            f"⚠️ Could not parse column-level drift dictionary: {err}"
        )
        drift_by_column = {}

    # 4. Weighted Feature Importance Matrix (SHAP / Domain Business Weights)
    feature_weights = {
        "lag_7": 0.25,
        "stock_on_hand": 0.20,
        "lag_14": 0.15,
        "lag_30": 0.10,
        "rolling_mean_7": 0.10,
        "rolling_mean_30": 0.10,
        "supplier_lead_time": 0.05,
        "month": 0.02,
        "dayofweek": 0.02,
        "is_weekend": 0.01,
    }

    # 5. Weighted Score Calculation
    total_weighted_drift = 0.0
    total_weight = sum(feature_weights.values())

    for col, weight in feature_weights.items():
        col_info = drift_by_column.get(col, {})
        is_drifting = (
            1.0
            if col_info.get("drift_detected", False)
            else 0.0
        )
        total_weighted_drift += is_drifting * weight

    weighted_drift_score = (
        total_weighted_drift / total_weight
        if total_weight > 0
        else 0.0
    )
    logging.info(
        f"Computed weighted drift score: {weighted_drift_score * 100:.1f}%"
    )

    # 6. Map Score to Operational Status
    if weighted_drift_score >= CRITICAL_THRESHOLD:
        status_value = "CRITICAL"
    elif weighted_drift_score >= WARNING_THRESHOLD:
        status_value = "WARNING"
    else:
        status_value = "STABLE"

    # 7. Persist State to Disk for API Consumption
    status_path = Path(STATUS_FILE_PATH)
    if status_path.is_dir():
        import shutil
        shutil.rmtree(status_path)
    with open(STATUS_FILE_PATH, "w") as f:
        f.write(status_value)

    logging.info(
        f"✅ Report saved -> {report_path} | Model Status Evaluated: {status_value}"
    )


if __name__ == "__main__":
    run_drift_monitoring()
