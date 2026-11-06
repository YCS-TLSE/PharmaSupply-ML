"""
Reference-data endpoints (stores and product catalogues).

RESPONSIBILITIES
----------------
* List the distinct store identifiers available in the staging table.
* List the product catalogue for a given store.
* Translate repository failures into explicit ``503`` responses.

INPUTS / OUTPUTS
----------------
Inputs: optional ``store_id`` path parameter.
Outputs: ``StoresOutput`` / ``ProductsOutput``. Empty datasets are returned as
empty lists (truthful), never replaced by fabricated identifiers.

ARCHITECTURE INTEGRATION
------------------------
The dashboard calls these endpoints to populate its selectors. Authentication
is enforced at router level so every route inherits the API-key dependency.
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
from src.repositories.inventory import (
    fetch_product_ids,
    fetch_store_ids,
)
from src.schemas import ProductsOutput, StoresOutput
from src.security import require_api_key
from src.services.inventory import normalize_store_id

# Router-level dependency: any future route added here is authenticated.
router = APIRouter(
    tags=["Data Fetch"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/stores", response_model=StoresOutput)
def get_stores() -> StoresOutput:
    """Return distinct store identifiers (empty list when no data exists).

    A database outage is surfaced as ``503`` so callers can distinguish
    "no stores" from "database unavailable".
    """
    try:
        store_ids = fetch_store_ids()
    except DatabaseUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable while listing stores.",
        ) from exc
    return StoresOutput(stores=store_ids)


@router.get(
    "/stores/{store_id}/products",
    response_model=ProductsOutput,
)
def get_products(store_id: str) -> ProductsOutput:
    """Return the product catalogue for a store.

    Store identifiers are normalised before the query so ``1``,
    ``STORE_001`` and ``STORE_1`` resolve to the same database rows.
    """
    candidates = normalize_store_id(store_id)
    try:
        product_ids = fetch_product_ids(candidates)
    except DatabaseUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable while listing products.",
        ) from exc
    return ProductsOutput(products=product_ids)
