"""
PharmaSupply-ML source package.

This package contains the production code of the pharmaceutical demand
forecasting platform:

* ``api.py`` / ``api_routers`` — FastAPI serving layer.
* ``services`` — pure business rules (alerts, replenishment, FEFO).
* ``repositories`` — pooled database access and SQL queries.
* ``features`` / ``train`` — feature engineering and XGBoost training.
* ``ingestion`` / ``validation`` — data loading and quality gates.
* ``monitoring`` / ``explainability`` — drift detection and SHAP reporting.
* ``dashboard.py`` — Streamlit decision dashboard.
* ``config.py`` — shared constants and environment-driven settings.

Keeping the package importable as ``src.*`` lets the pipeline, the API and
the tests share exactly the same implementation.
"""
