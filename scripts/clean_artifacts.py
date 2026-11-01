"""
===============================================================================
PharmaSupply-ML: Environment & Artifact Cleanup Utility
===============================================================================

OBJECTIVE & USAGE:
------------------
This script decouples the environment reset and artifact wiping phase from the
master pipeline orchestrator. It allows engineers to safely clean previous run
artifacts (models, reports, figures) and reset the model health status without
risking Docker volume mount corruption.

Usage via CLI:
    python scripts/clean_artifacts.py          # Standard run artifact cleanup
    python scripts/clean_artifacts.py --all    # Comprehensive cleanup (including figures)
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clean_artifacts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def clean_artifacts(
    reset_status: bool = True,
    include_figures: bool = False,
    verbose: bool = True,
) -> None:
    """Safely cleans previous run artifacts and resets model status.

    Parameters:
        reset_status (bool): Whether to reset model_status.txt to 'STABLE'.
        include_figures (bool): Whether to remove SHAP figures in reports/figures/.
        verbose (bool): Whether to log detailed cleanup messages.
    """
    if verbose:
        logger.info("Initiating artifact cleanup...")

    # 1. Reset model_status.txt to nominal state (NEVER delete file to protect Docker bind mounts)
    if reset_status:
        status_file = PROJECT_ROOT / "model_status.txt"
        if status_file.is_dir():
            shutil.rmtree(status_file)
        status_file.write_text("STABLE\n", encoding="utf-8")
        if verbose:
            logger.info("   - Reset 'model_status.txt' to nominal state ('STABLE')")

    # 2. Files to remove
    target_files = [
        PROJECT_ROOT / "reports" / "feature_weights.json",
        PROJECT_ROOT / "reports" / "drift_report.html",
        PROJECT_ROOT / "models" / "xgboost_pharma_demand.joblib",
    ]

    if include_figures:
        figures_dir = PROJECT_ROOT / "reports" / "figures"
        if figures_dir.exists():
            for fig in figures_dir.glob("*"):
                if fig.is_file():
                    target_files.append(fig)

    for file_path in target_files:
        if file_path.exists():
            try:
                file_path.unlink()
                if verbose:
                    try:
                        rel_path = file_path.relative_to(PROJECT_ROOT)
                    except ValueError:
                        rel_path = file_path
                    logger.info(f"   - Removed artifact: {rel_path}")
            except Exception as e:
                logger.warning(f"   - Could not remove {file_path}: {e}")

    if verbose:
        logger.info("Artifact cleanup successfully completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean previous PharmaSupply-ML run artifacts and reset environment."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include generated SHAP figures in reports/figures/.",
    )
    parser.add_argument(
        "--no-reset-status",
        action="store_true",
        help="Do not reset model_status.txt to STABLE.",
    )
    args = parser.parse_args()

    clean_artifacts(
        reset_status=not args.no_reset_status,
        include_figures=args.all,
        verbose=True,
    )


if __name__ == "__main__":
    main()

