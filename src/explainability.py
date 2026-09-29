"""
Phase 05: Model Explainability Module (PharmaSupply-ML).

This module provides an industrialization wrapper for model interpretability
using SHAP (SHapley Additive exPlanations). It extracts global feature importance
and local prediction justifications for the trained XGBoost demand forecasting model,
and persists resulting plots for reporting and dashboarding.

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

from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv


class ModelExplainer:
    """Handles SHAP-based global and local interpretability for tree-based models."""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "reports/figures",
    ):
        """
        Initialize the explainer with model artifact path and output destination.

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
        """
        Load the trained model artifact from disk.

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
        """
        Initialize the SHAP TreeExplainer and compute SHAP values.
        Applies dynamic subsampling and optional multi-pass aggregation for sampling variance reduction.

        Parameters:
            X_sample (pd.DataFrame): Feature matrix provided by pipeline or user upload.
            threshold (int): Minimum row count to trigger subsampling (default: 300).
            sample_size (int): Number of rows to sample per pass (default: 200).
            n_repeats (int): Number of sampling passes. 1 = Fast Prod Mode, >1 = Robust Audit Mode.
        """
        # TreeExplainer is specifically optimized for tree-based architectures like XGBoost
        self.explainer = shap.TreeExplainer(self.model)

        # 1. Case where input dataset is small: process fully without subsampling
        if len(X_sample) <= threshold:
            print(
                f"[INFO] Dataset size ({len(X_sample)}) <= {threshold}. "
                "Using full dataset for SHAP calculation."
            )
            X_eval = X_sample
            self.shap_values = self.explainer(X_eval)

        # 2. Case where input dataset exceeds threshold: trigger subsampling logic
        else:
            if n_repeats > 1:
                # Robust Audit Mode: Average SHAP values across multiple sampling iterations
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
                        X_eval = X_sub  # Keep reference sample for evaluation matrix

                    shap_accumulated.append(
                        self.explainer(X_sub).values
                    )

                # Compute mean SHAP values across all passes to eliminate single-draw bias
                mean_shap_values = np.mean(
                    shap_accumulated, axis=0
                )

                # Construct a unified SHAP Explanation object
                self.shap_values = shap.Explanation(
                    values=mean_shap_values,
                    base_values=self.explainer(
                        X_eval
                    ).base_values,
                    data=X_eval.values,
                    feature_names=X_eval.columns,
                )
            else:
                # Fast Prod Mode: Deterministic single pass for low-latency inference
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
        """
        Generate and save global explainability plots:
        1. Summary Beeswarm plot: Distribution of feature impacts across observations.
        2. Bar plot: Global ranking of mean absolute SHAP feature importance.

        Parameters:
            max_display (int): Top features to display to avoid graphical noise (default: 10).
        """
        if (
            self.explainer is None
            or self.shap_values is None
        ):
            raise ValueError(
                "Explainer is not initialized. Run fit_explainer() first."
            )

        # 1. Beeswarm Plot (Feature impact distribution and directionality)
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

        # 2. Bar Plot (Mean global feature importance ranking)
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
        """
        Generate and save an Individual Breakdown Bar Plot explaining a single prediction instance.

        Parameters:
            sample_index (int): Row index in the target matrix to explain.
            max_display (int): Top features to display for the breakdown (default: 10).
            filename (str): Output filename for the generated plot artifact.
        """
        if (
            self.explainer is None
            or self.shap_values is None
        ):
            raise ValueError(
                "Explainer is not initialized. Run fit_explainer() first."
            )

        # Local Breakdown Bar plot: Deconstructs base value E[f(x)] to final prediction f(x)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(
            self.shap_values[sample_index],
            max_display=max_display,
            show=False,
        )

        # Calculate f(x) and baseline for detailed title annotation
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
        """
        Export the computed SHAP values matrix as a CSV file for downstream usage.

        Parameters:
            filename (str): Target CSV filename in the output directory.
        """
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


if __name__ == "__main__":
    load_dotenv()

    # 1. Chargement des données brutes
    data_path = "data/pharmaceutical_demand_row.csv"
    df = pd.read_csv(data_path)

    # 2. Feature Engineering à la volée (calcul des colonnes requises par XGBoost)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = (
            df["dayofweek"].isin([5, 6]).astype(int)
        )

    # Identification de la colonne cible (target)
    target_col = (
        "target_demand"
        if "target_demand" in df.columns
        else "demand"
    )

    if target_col in df.columns:
        df["lag_7"] = df[target_col].shift(7)
        df["lag_14"] = df[target_col].shift(14)
        df["lag_30"] = df[target_col].shift(30)
        df["rolling_mean_7"] = (
            df[target_col].shift(1).rolling(7).mean()
        )
        df["rolling_mean_30"] = (
            df[target_col].shift(1).rolling(30).mean()
        )

    # Nettoyage des lignes avec des valeurs manquantes dues aux décalages (lags)
    df = df.dropna().reset_index(drop=True)

    # 3. Initialisation de l'explainer et alignement des colonnes
    explainer = ModelExplainer(
        model_path="models/xgboost_pharma_demand.joblib"
    )

    if hasattr(explainer.model, "feature_names_in_"):
        expected_features = list(
            explainer.model.feature_names_in_
        )
        X = df[expected_features]
    else:
        X = df.drop(
            columns=[target_col, "date", "product_id"],
            errors="ignore",
        ).select_dtypes(include=[np.number])

    # 4. Calcul des valeurs SHAP et génération des artefacts
    explainer.fit_explainer(
        X, threshold=300, sample_size=200, n_repeats=1
    )
    explainer.generate_global_plots(max_display=10)
    explainer.generate_local_breakdown(
        sample_index=0, max_display=10
    )
    explainer.export_shap_to_csv()
