"""
===============================================================================
PharmaSupply-ML: Data Drift Monitoring
===============================================================================

PURPOSE
-------
Measure distribution drift and translate it into an operational model status
consumed by the API health endpoints.

RESPONSIBILITIES
----------------
* Offline mode: compare an 80/20 chronological holdout of historical data.
* Production mode: compare the reference distribution with aggregated features
  actually observed on ``/predict`` (JSONL prediction log).
* Load normalised SHAP weights from ``reports/feature_weights.json``.
* Compute a weighted drift score and map it to STABLE/WARNING/CRITICAL.
* Persist the status atomically for the API watcher.

INPUTS / OUTPUTS
----------------
Inputs: feature DataFrames, ``reports/feature_weights.json`` and
``reports/prediction_log.jsonl``.
Outputs: Evidently HTML reports and ``model_status.txt``.

ARCHITECTURE INTEGRATION
------------------------
Invoked by ``run_pipeline.py`` and batch schedulers. The API only reads the
resulting status file; it never runs Evidently in the request path.

OPERATIONAL NOTES
-----------------
Production drift returns ``UNKNOWN`` until enough prediction samples exist.
Malformed log lines are skipped so a partial write cannot break the batch.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import (
    DEFAULT_FEATURE_WEIGHTS,
    DRIFT_CRITICAL_THRESHOLD,
    DRIFT_WARNING_THRESHOLD,
    FEATURE_COLS,
    FEATURE_WEIGHTS_PATH,
    PRODUCTION_DRIFT_MIN_SAMPLES,
    STATUS_FILE_PATH,
    TARGET_COL,
)
from src.observability import read_prediction_log
from src.train import create_features, load_validated_data

logger = logging.getLogger("pharmasupply.monitoring")


# Load the SHAP-derived weights. A missing/corrupt artifact falls back to the
# documented defaults so monitoring remains operational.
def load_feature_weights(
    path: str | Path | None = None,
) -> dict[str, float]:
    """Load normalised SHAP weights, falling back to documented defaults."""
    weights_path = Path(path or FEATURE_WEIGHTS_PATH)
    if not weights_path.exists():
        logger.warning(
            "Feature weights not found at '%s'; using defaults.",
            weights_path,
        )
        return dict(DEFAULT_FEATURE_WEIGHTS)

    try:
        raw = json.loads(
            weights_path.read_text(encoding="utf-8")
        )
        weights = {
            str(column): float(weight)
            for column, weight in raw.items()
            if str(column) in FEATURE_COLS
        }
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            "Unable to read feature weights (%s); using defaults.",
            exc,
        )
        return dict(DEFAULT_FEATURE_WEIGHTS)

    return weights or dict(DEFAULT_FEATURE_WEIGHTS)


# Weighted share of drifting features. Domain-important features (lags, stock)
# contribute more than calendar attributes.
def compute_weighted_drift_score(
    drift_by_column: Mapping[str, Mapping[str, Any]],
    feature_weights: Mapping[str, float],
) -> float:
    """Weighted share of drifting features (0.0 to 1.0)."""
    total_weight = sum(feature_weights.values())
    if total_weight <= 0:
        return 0.0

    drifting_weight = 0.0
    for column, weight in feature_weights.items():
        column_info = drift_by_column.get(column) or {}
        if column_info.get("drift_detected", False):
            drifting_weight += weight
    return drifting_weight / total_weight


# Threshold mapping centralised so batch and API agree on status semantics.
def map_drift_status(score: float) -> str:
    """Map a weighted drift score to an operational status."""
    if score >= DRIFT_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= DRIFT_WARNING_THRESHOLD:
        return "WARNING"
    return "STABLE"


# Atomic status persistence: write a temporary file then os.replace, so the API
# can never read a partially written status.
def _write_status(status_value: str) -> None:
    """Persist the status atomically so the API never reads a partial file."""
    status_path = Path(STATUS_FILE_PATH)
    if status_path.is_dir():
        raise RuntimeError(
            f"Status path '{status_path}' is a directory; refusing to "
            "delete it. Fix STATUS_FILE_PATH."
        )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_name(
        status_path.name + ".tmp"
    )
    tmp_path.write_text(status_value, encoding="utf-8")
    os.replace(tmp_path, status_path)


# Evidently stores column-level flags inside a nested metrics dictionary. This
# helper isolates the fragile parsing and tolerates version differences.
def _extract_drift_by_column(
    evaluation: Any,
) -> dict[str, Any]:
    """Extract column-level drift flags from an Evidently evaluation."""
    try:
        result_dict = (
            evaluation.as_dict()
            if hasattr(evaluation, "as_dict")
            else evaluation.dict()
        )
        for metric in result_dict.get("metrics", []):
            if "DatasetDriftMetric" in metric.get(
                "metric", ""
            ):
                return metric.get("result", {}).get(
                    "drift_by_columns", {}
                )
    except (
        Exception
    ) as exc:  # pragma: no cover - Evidently internals
        logger.warning(
            "Could not parse column-level drift dictionary: %s",
            exc,
        )
    return {}


# Shared Evidently execution: run the preset, optionally save HTML, and return
# column-level drift flags.
def _run_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: str | None = None,
) -> dict[str, Any]:
    """Run an Evidently DataDriftPreset and optionally save its HTML report."""
    report = Report(metrics=[DataDriftPreset()])
    evaluation = report.run(
        reference_data=reference_df, current_data=current_df
    )

    if output_html_path:
        report_path = Path(output_html_path)
        report_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        target = (
            evaluation if evaluation is not None else report
        )
        if hasattr(target, "save_html"):
            target.save_html(str(report_path))
        elif hasattr(target, "save"):
            target.save(str(report_path))

    return _extract_drift_by_column(
        evaluation if evaluation is not None else report
    )


# Offline holdout check used by the pipeline as a regression signal.
def run_drift_monitoring(
    df: pd.DataFrame = None,
    output_html_path: str = "reports/drift_report.html",
) -> None:
    """Offline chronological holdout drift check (pipeline regression)."""
    if df is None:
        logger.info(
            "Fetching raw dataset from PostgreSQL database..."
        )
        df = create_features(load_validated_data())

    unique_dates = df["date"].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    model_columns = list(FEATURE_COLS) + [TARGET_COL]
    reference_df = df[df["date"] < split_date][
        model_columns
    ]
    current_df = df[df["date"] >= split_date][model_columns]

    drift_by_column = _run_report(
        reference_df, current_df, output_html_path
    )
    score = compute_weighted_drift_score(
        drift_by_column, load_feature_weights()
    )
    status_value = map_drift_status(score)
    _write_status(status_value)

    logger.info(
        "Offline drift score: %.1f%% | status: %s | report: %s",
        score * 100,
        status_value,
        output_html_path,
    )


# Production drift: compare live prediction features with the training
# reference. Returns UNKNOWN when the sample size is insufficient.
def run_production_drift_monitoring(
    prediction_log_path: str | Path | None = None,
    reference_df: pd.DataFrame | None = None,
    output_html_path: str = "reports/production_drift_report.html",
) -> str:
    """Compare live prediction features with the training reference.

    Returns the computed status, or ``UNKNOWN`` when there are not enough
    production records to make a statistically meaningful comparison.
    """
    records = read_prediction_log(prediction_log_path)
    if len(records) < PRODUCTION_DRIFT_MIN_SAMPLES:
        logger.warning(
            "Not enough production records for drift (%d < %d).",
            len(records),
            PRODUCTION_DRIFT_MIN_SAMPLES,
        )
        return "UNKNOWN"

    current_df = pd.DataFrame(records)
    missing_columns = [
        column
        for column in FEATURE_COLS
        if column not in current_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Prediction log is missing feature columns: "
            f"{missing_columns}"
        )
    current_df = current_df[list(FEATURE_COLS)]

    if reference_df is None:
        reference_df = create_features(
            load_validated_data()
        )[list(FEATURE_COLS)]
    else:
        reference_df = reference_df[list(FEATURE_COLS)]

    drift_by_column = _run_report(
        reference_df, current_df, output_html_path
    )
    score = compute_weighted_drift_score(
        drift_by_column, load_feature_weights()
    )
    status_value = map_drift_status(score)
    _write_status(status_value)

    logger.info(
        "Production drift score: %.1f%% | status: %s | samples: %d",
        score * 100,
        status_value,
        len(current_df),
    )
    return status_value


if __name__ == "__main__":
    run_drift_monitoring()
