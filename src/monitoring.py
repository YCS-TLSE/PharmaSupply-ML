"""
===============================================================================
PharmaSupply-ML: Advanced MLOps Data Drift Monitoring Module
===============================================================================

OBJECTIVE & USAGE:
------------------
This script automates the production monitoring pipeline for the pharmaceutical
demand forecasting model. It serves a dual purpose:
1. Operational MLOps: Evaluates data distribution stability over time by
   comparing historical reference data against recent production streams using
   Evidently AI.
2. Business-Centric Risk Mitigation: Unlike naive monitoring tools that weigh
   all columns equally, this module computes a **weighted drift score** based
   on feature importance and critical business weights (derived from SHAP values
   and domain logic).

WORKFLOW & ARCHITECTURE:
------------------------
1. Data Ingestion & Splitting: Automatically fetches historical records from
   PostgreSQL and splits them chronologically into reference (baseline) and
   current production subsets.
2. Statistical Drift Evaluation: Generates an Evidently AI DataDriftPreset report
   and maps column-level drift outcomes.
3. Weighted Criticality Scoring: Multiplies feature-level drift flags by predefined
   business/SHAP weights (e.g., lag_7 and stock_on_hand carry much higher weights
   than calendar attributes).
4. Status Dispatch & Alerting:
   - STABLE: Nominal behavior, reliable predictions.
   - WARNING: Moderate drift on secondary features; lightweight caution advised.
   - CRITICAL: Severe drift on critical business drivers; triggers an immediate
     SMTP alert email and switches dashboard UI safety flags.
5. State Persistence: Writes the final status string into a shared persistence file
   ('model_status.txt') consumed downstream by the FastAPI backend and Streamlit dashboard.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.train import create_features, load_validated_data

# Configure standard logging format for tracking automated batch executions
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define feature columns tracked by the forecasting model
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

# Path configuration for model health status communication across microservices
STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH", "model_status.txt"
)

# Threshold definitions for the weighted drift score
WARNING_THRESHOLD = 0.15  # 15% weighted drift -> WARNING status (moderate risk)
CRITICAL_THRESHOLD = 0.30  # 30% weighted drift -> CRITICAL status + Email alert


def send_drift_alert_email(weighted_drift_score: float):
    """
    Sends an automated alert email to the Data/MLOps engineering team
    when a critical data drift score is breached.

    Parameters:
        weighted_drift_score (float): The computed weighted drift percentage.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    sender_email = os.getenv(
        "SENDER_EMAIL", "mlops-alert@pharmasupply.com"
    )
    receiver_email = os.getenv(
        "RECEIVER_EMAIL", "data-team@pharmasupply.com"
    )
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    # Skip sending if SMTP credentials are missing (local testing mode)
    if not smtp_password:
        logging.warning(
            "⚠️ No SMTP password configured. Alert email dispatch skipped."
        )
        return

    # Construct email payload
    msg = EmailMessage()
    msg["Subject"] = (
        "🚨 [MLOps Alert] Critical Data Drift Detected on Key Features"
    )
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(
        f"Attention MLOps Team,\n\n"
        f"The automated nightly batch detected a critical weighted drift score: {weighted_drift_score * 100:.1f}%.\n"
        f"The model health status has been switched to 'CRITICAL'. Immediate inspection of feature pipelines is required.\n"
    )

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(msg)
        logging.info(
            "📧 Critical alert email successfully dispatched."
        )
    except Exception as e:
        logging.error(
            f"❌ Failed to dispatch alert email: {e}"
        )


def run_drift_monitoring(
    df: pd.DataFrame = None,
    output_html_path: str = "reports/drift_report.html",
) -> None:
    """
    Executes the full data drift analysis pipeline, calculates the weighted
    criticality score, updates status persistence, and triggers alerts if needed.

    Parameters:
        df (pd.DataFrame, optional): Pre-loaded feature dataframe. If None, fetches from DB.
        output_html_path (str): Destination path for the generated Evidently HTML report.
    """
    # 1. Data Retrieval & Preparation
    if df is None:
        logging.info(
            "Fetching raw dataset from PostgreSQL database..."
        )
        df_raw = load_validated_data()
        df = create_features(df_raw)

    # Chronological splitting (80% reference historical baseline, 20% current production)
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

    # Ensure output directory exists and persist report to disk
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

    # 3. Column-Level Drift Extraction
    try:
        eval_obj = (
            my_eval if my_eval is not None else report
        )
        result_dict = (
            eval_obj.as_dict()
            if hasattr(eval_obj, "as_dict")
            else eval_obj.dict()
        )

        # Isolate column-level evaluation metrics from Evidently's dictionary structure
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

    # 4. Weighted Feature Importance Definition (Derived from SHAP / Domain Business Logic)
    feature_weights = {
        "lag_7": 0.25,  # High impact on short-term demand forecasting
        "stock_on_hand": 0.20,  # Critical operational constraint variable
        "lag_14": 0.15,  # Secondary temporal trend
        "lag_30": 0.10,  # Long-term baseline trend
        "rolling_mean_7": 0.10,  # Smoothed short-term momentum
        "rolling_mean_30": 0.10,  # Smoothed baseline momentum
        "supplier_lead_time": 0.05,  # Operational supply constraint
        "month": 0.02,  # Seasonal calendar indicator
        "dayofweek": 0.02,  # Weekly cyclical indicator
        "is_weekend": 0.01,  # Binary weekend marker
    }

    # 5. Compute Weighted Drift Score
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

    # 6. Map Score to the 3 Explicit Production States & Dispatch Actions
    if weighted_drift_score >= CRITICAL_THRESHOLD:
        status_value = "CRITICAL"
        send_drift_alert_email(weighted_drift_score)
    elif weighted_drift_score >= WARNING_THRESHOLD:
        status_value = "WARNING"
    else:
        status_value = "STABLE"

    # 7. Persist Status State for Downstream API Consumption
    with open(STATUS_FILE_PATH, "w") as f:
        f.write(status_value)

    logging.info(
        f"✅ Report successfully saved -> {report_path} | Evaluated Model Status: {status_value}"
    )


if __name__ == "__main__":
    run_drift_monitoring()
