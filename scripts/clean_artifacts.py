"""
===============================================================================
PharmaSupply-ML: Artifact Cleanup Utility
===============================================================================

PURPOSE
-------
Safely remove previous run artifacts and reset the model health status without
breaking Docker bind mounts.

RESPONSIBILITIES
----------------
* Reset ``model_status.txt`` to ``STABLE`` (never delete the file, because it is
  bind-mounted into containers).
* Remove the model artifact, drift report and feature weights.
* Optionally remove generated SHAP figures.

INPUTS / OUTPUTS
----------------
Inputs: CLI flags (``--all``, ``--no-reset-status``) or the orchestrator call.
Outputs: deleted files and a reset status file; logs describe each action.

ARCHITECTURE INTEGRATION
------------------------
Invoked by ``run_pipeline.py --clean`` and directly from the CLI. The
bind-mount protection is essential: deleting the status file would make Docker
create a directory at that path on the next start.
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


# Public cleanup routine shared by the CLI and the orchestrator.
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
            logger.info(
                "   - Reset 'model_status.txt' to nominal state ('STABLE')"
            )

    # 2. Files to remove
    target_files = [
        PROJECT_ROOT / "reports" / "feature_weights.json",
        PROJECT_ROOT / "reports" / "drift_report.html",
        PROJECT_ROOT
        / "models"
        / "xgboost_pharma_demand.joblib",
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
                        rel_path = file_path.relative_to(
                            PROJECT_ROOT
                        )
                    except ValueError:
                        rel_path = file_path
                    logger.info(
                        f"   - Removed artifact: {rel_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"   - Could not remove {file_path}: {e}"
                )

    if verbose:
        logger.info(
            "Artifact cleanup successfully completed."
        )


# CLI wrapper: translates flags into the cleanup options.
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
