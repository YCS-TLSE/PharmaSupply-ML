"""
===============================================================================
Phase 05: Model Explainability & Feature Weight Export Module (PharmaSupply-ML)
===============================================================================

OBJECTIVE & USAGE:
------------------
This module provides an industrialization wrapper for model interpretability
using SHAP (SHapley Additive exPlanations). It serves a dual purpose:
1. Model Transparency & Reporting: Extracts global feature importance and local
   prediction justifications for the trained XGBoost demand forecasting model,
   persisting resulting plots for dashboards and stakeholder audits.
2. MLOps Monitoring Bridge: Computes mean absolute SHAP values, normalizes them
   into percentage-based feature importance weights, and exports them to a JSON
   artifact. This artifact is consumed downstream by 'monitoring.py' to drive
   weighted data drift calculations.

Design & Performance Strategy:
------------------------------
1. Dynamic Subsampling (Threshold-based):
   Computing SHAP values on large datasets (> 300 rows) causes visual clutter
   (overlapping dots in beeswarm plots) and latency bottlenecks in production.
   This module dynamically subsamples inputs exceeding the threshold.

2. Single-Pass vs. Aggregated Multi-Pass Sampling (Cross-Validation Sampling):
   - n_repeats = 1 (Fast Prod Mode): Uses a deterministic single-draw sample
     (random_state=42). Provides fast inference (< 0.3s) for REST API and Streamlit.
   - n_repeats > 1 (Robust Audit Mode): Computes SHAP values across 'n_repeats'
     independent random samples and averages them. Reduces sampling variance
     for official reporting and high-precision audit requirements.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv

# Direct import of validated Feature Engineering and database loading functions
from src.train import create_features, load_validated_data

# ==============================================================================
# ROBUST MONKEY-PATCH FOR SHAP & XGBOOST >= 3.0 COMPATIBILITY
# ==============================================================================
# Intercepts SHAP's internal tree model loader to safely clean bracketed base_score strings
try:
    import shap.explainers._tree as shap_tree

    if hasattr(shap_tree, "XGBTreeModelLoader"):
        _original_xgb_loader_init = (
            shap_tree.XGBTreeModelLoader.__init__
        )

        def _patched_xgb_loader_init(self, model):
            _original_xgb_loader_init(self, model)
            if hasattr(
                self, "learner_model_param"
            ) and isinstance(
                self.learner_model_param, dict
            ):
                bs = self.learner_model_param.get(
                    "base_score"
                )
                if bs is not None:
                    if isinstance(bs, str):
                        cleaned = bs.strip("[]'\" ")
                        try:
                            self.learner_model_param[
                                "base_score"
                            ] = float(cleaned)
                        except ValueError:
                            self.learner_model_param[
                                "base_score"
                            ] = 0.5
                    elif isinstance(bs, list):
                        try:
                            self.learner_model_param[
                                "base_score"
                            ] = float(bs[0])
                        except (ValueError, IndexError):
                            self.learner_model_param[
                                "base_score"
                            ] = 0.5

        shap_tree.XGBTreeModelLoader.__init__ = (
            _patched_xgb_loader_init
        )
except Exception:
    pass


class ModelExplainer:
    """Handles SHAP-based global/local interpretability and automated feature weighting for tree models."""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "reports/figures",
    ):
        """Initializes the explainer with model artifact path and output destination.

        Parameters:
            model_path (str): Path to the serialized joblib model artifact.
            output_dir (str): Directory where generated figures will be stored.
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[shap.Explanation] = None
        self.X_sample: Optional[pd.DataFrame] = None

    def _load_model(self):
        """Loads the trained model artifact from disk.

        Returns:
            object: Loaded XGBoost model artifact.

        Raises:
            FileNotFoundError: If the model file is missing at the specified path.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at path: {self.model_path}"
            )
        return joblib.load(self.model_path)

    def fit_explainer(
        self,
        X_sample: pd.DataFrame,
        threshold: int = 300,
        sample_size: int = 200,
        n_repeats: int = 1,
    ):
        """Initializes the SHAP TreeExplainer and computes SHAP values.

        Applies dynamic subsampling and optional multi-pass aggregation for
        sampling variance reduction.

        Parameters:
            X_sample (pd.DataFrame): Feature matrix provided by pipeline or user upload.
            threshold (int): Minimum row count to trigger subsampling (default: 300).
            sample_size (int): Number of rows to sample per pass (default: 200).
            n_repeats (int): Number of sampling passes. 1 = Fast Prod Mode, >1 = Robust Audit Mode.
        """
        # Extract native booster
        booster = (
            self.model.get_booster()
            if hasattr(self.model, "get_booster")
            else self.model
        )

        self.explainer = shap.TreeExplainer(booster)

        if len(X_sample) <= threshold:
            print(
                f"[INFO] Dataset size ({len(X_sample)}) <= {threshold}. "
                "Using full dataset for SHAP calculation."
            )
            X_eval = X_sample
            self.shap_values = self.explainer(X_eval)
        else:
            if n_repeats > 1:
                print(
                    f"[INFO] Dataset size ({len(X_sample)}) > {threshold}. "
                    f"Running {n_repeats}-pass aggregated SHAP calculation (sample_size={sample_size})..."
                )
                shap_accumulated = []
                X_eval = None

                for i in range(n_repeats):
                    X_sub = X_sample.sample(
                        n=sample_size, random_state=42 + i
                    )
                    if i == 0:
                        X_eval = X_sub

                    shap_accumulated.append(
                        self.explainer(X_sub).values
                    )

                mean_shap_values = np.mean(
                    shap_accumulated, axis=0
                )

                self.shap_values = shap.Explanation(
                    values=mean_shap_values,
                    base_values=self.explainer(
                        X_eval
                    ).base_values,
                    data=X_eval.values,
                    feature_names=X_eval.columns,
                )
            else:
                print(
                    f"[INFO] Dataset size ({len(X_sample)}) > {threshold}. "
                    f"Subsampled to {sample_size} rows (single-pass, random_state=42)."
                )
                X_eval = X_sample.sample(
                    n=sample_size, random_state=42
                )
                self.shap_values = self.explainer(X_eval)

        self.X_sample = X_eval

    def generate_global_plots(self, max_display: int = 10):
        """Generates and saves global explainability visual artifacts."""
        if (
            self.explainer is None
            or self.shap_values is None
        ):
            raise ValueError(
                "Explainer is not initialized. Run fit_explainer() first."
            )

        # 1. Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(
            self.shap_values,
            max_display=max_display,
            show=False,
        )
        beeswarm_path = (
            self.output_dir / "shap_summary_beeswarm.png"
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=300)
        plt.close()

        # 2. Bar Plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(
            self.shap_values,
            max_display=max_display,
            show=False,
        )
        bar_path = self.output_dir / "shap_summary_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        print(
            f"[OK] Global SHAP plots successfully saved to: {self.output_dir}"
        )

    def generate_local_breakdown(
        self,
        sample_index: int = 0,
        max_display: int = 10,
        filename: str = "shap_individual_breakdown.png",
    ):
        """Generates and saves an individual prediction breakdown bar plot."""
        if (
            self.explainer is None
            or self.shap_values is None
        ):
            raise ValueError(
                "Explainer is not initialized. Run fit_explainer() first."
            )

        plt.figure(figsize=(10, 6))
        shap.plots.bar(
            self.shap_values[sample_index],
            max_display=max_display,
            show=False,
        )

        f_x = self.shap_values[
            sample_index
        ].base_values + np.sum(
            self.shap_values[sample_index].values
        )
        baseline = self.shap_values[
            sample_index
        ].base_values

        plt.title(
            f"Individual Demand Prediction Breakdown (f(x) = {f_x:.2f} units)\n"
            f"Baseline E[f(X)] = {baseline:.2f} units",
            fontsize=11,
        )

        breakdown_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(breakdown_path, dpi=300)
        plt.close()

        print(
            f"[OK] Local explanation plot (index {sample_index}) saved to: {breakdown_path}"
        )

    def export_shap_to_csv(
        self, filename: str = "shap_values.csv"
    ):
        """Exports the computed SHAP values matrix as a tabular CSV file."""
        if (
            self.shap_values is None
            or self.X_sample is None
        ):
            raise ValueError(
                "No SHAP values computed. Run fit_explainer() first."
            )

        df_shap = pd.DataFrame(
            self.shap_values.values,
            columns=self.X_sample.columns,
            index=self.X_sample.index,
        )
        csv_path = self.output_dir / filename
        df_shap.to_csv(csv_path)
        print(f"[OK] SHAP matrix exported to: {csv_path}")

    def export_feature_weights(
        self, filename: str = "feature_weights.json"
    ):
        """Computes mean absolute SHAP values as normalized feature importance weights and exports to JSON.

        This artifact bridges the explainability layer with the MLOps monitoring module.
        """
        if (
            self.shap_values is None
            or self.X_sample is None
        ):
            raise ValueError(
                "No SHAP values computed. Run fit_explainer() first."
            )

        # Calculate mean absolute SHAP value per feature to represent real global impact
        mean_abs_shap = np.mean(
            np.abs(self.shap_values.values), axis=0
        )
        weights_raw = dict(
            zip(self.X_sample.columns, mean_abs_shap)
        )

        # Normalize weights to sum up to 1.0 (percentages for weighted drift calculation)
        total_shap = sum(weights_raw.values())
        normalized_weights = (
            {
                col: float(val / total_shap)
                for col, val in weights_raw.items()
            }
            if total_shap > 0
            else weights_raw
        )

        weights_path = self.output_dir.parent / filename
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(normalized_weights, f, indent=4)

        print(
            f"[OK] Normalized feature weights successfully exported to: {weights_path}"
        )


if __name__ == "__main__":
    load_dotenv()

    # 1. Load historical records and build features via database pipeline
    df_raw = load_validated_data()
    df_features = create_features(df_raw)

    # 2. Initialize the ModelExplainer wrapper
    explainer = ModelExplainer(
        model_path="models/xgboost_pharma_demand.joblib"
    )

    # 3. Strict feature alignment against model expectations
    if hasattr(explainer.model, "feature_names_in_"):
        expected_features = list(
            explainer.model.feature_names_in_
        )
        X = df_features[expected_features]
    else:
        X = df_features.drop(
            columns=[
                "target_demand",
                "date",
                "product_id",
                "store_id",
            ],
            errors="ignore",
        ).select_dtypes(include=[np.number])

    # 4. Compute SHAP values, generate reports, and export monitoring weights
    explainer.fit_explainer(
        X, threshold=300, sample_size=200, n_repeats=1
    )
    explainer.generate_global_plots(max_display=10)
    explainer.generate_local_breakdown(
        sample_index=0, max_display=10
    )
    explainer.export_shap_to_csv()
    explainer.export_feature_weights()
