"""
Shared pytest fixtures and import-path bootstrap.

The project is executed from the repository root (``python -m pytest``), but
individual test modules may be run from other directories. Adding the project
root to ``sys.path`` here keeps ``src.*`` imports stable without repeating the
bootstrap in every file.

``override_model`` replaces the FastAPI model dependency with a deterministic
stub, allowing inference tests to run without loading the real joblib artifact.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def override_model():
    """Override the /predict model dependency for deterministic tests."""
    from src.api import app
    from src.api_routers.predict import get_loaded_model

    def _override(model):
        app.dependency_overrides[get_loaded_model] = (
            lambda: model
        )

    yield _override
    app.dependency_overrides.clear()
