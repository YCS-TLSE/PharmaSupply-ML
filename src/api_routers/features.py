"""
Feature retrieval endpoint used by the dashboard and prediction clients.

RESPONSIBILITIES
----------------
* Fetch the most recent inventory rows for a store/product pair.
* Build the typed ``FeaturesOutput`` payload, including the chronological
  ``sales_history`` used by the parity-safe inference path.
* Distinguish "no history" (``404``) from "database unavailable" (``503``).

INPUTS / OUTPUTS
----------------
Inputs: ``store_id`` and ``product_id`` path parameters.
Outputs: ``FeaturesOutput`` with logistical constraints, lags, FEFO expiry and
a 30-day sales history.

ARCHITECTURE INTEGRATION
------------------------
The dashboard forwards ``sales_history`` to ``POST /predict`` so the API can
use the same feature builder as training, eliminating train/serve skew.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.repositories.database import (
    DatabaseUnavailableError,
)
from src.repositories.inventory import fetch_feature_rows
from src.schemas import FeaturesOutput
from src.security import require_api_key
from src.services.inventory import (
    features_from_rows,
    normalize_store_id,
)

router = APIRouter(
    tags=["Data Fetch"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "/features/{store_id}/{product_id}",
    response_model=FeaturesOutput,
)
def get_latest_features(
    store_id: str, product_id: str
) -> FeaturesOutput:
    """Return recent features and a 30-day sales history for a product."""
    candidates = normalize_store_id(store_id)
    try:
        rows = fetch_feature_rows(candidates, product_id)
    except DatabaseUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable while fetching product features.",
        ) from exc

    # No history is a legitimate 404; returning synthetic defaults would make
    # the dashboard display plausible but false product data.
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No history found for store '{store_id}' "
                f"and product '{product_id}'."
            ),
        )

    # Service layer converts raw rows into the typed contract and computes the
    # inference-side features with the shared formulas.
    return features_from_rows(rows)
