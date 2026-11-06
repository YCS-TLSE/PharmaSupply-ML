"""
Inventory audit and multi-level alert endpoint.

RESPONSIBILITIES
----------------
* Read the latest inventory row per product for a store.
* Delegate alert classification to ``src.services.inventory.build_alerts``.
* Enrich the result with the current model drift status (level-3 alert).

INPUTS / OUTPUTS
----------------
Inputs: ``store_id`` path parameter.
Outputs: ``StoreAlertsOutput`` with alerts sorted by level then severity.
Database failures produce ``503``; the endpoint never returns a partial audit
that could be mistaken for a healthy inventory.

OPERATIONAL NOTES
-----------------
This route is rate-limited because it scans inventory rows and can be called
repeatedly by dashboard reruns.
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

import src.config as config
from src.repositories.database import (
    DatabaseUnavailableError,
)
from src.repositories.inventory import fetch_alert_rows
from src.schemas import StoreAlertsOutput
from src.security import limiter, require_api_key
from src.services.inventory import (
    build_alerts,
    normalize_store_id,
)
from src.services.model_status import read_model_status

# Authentication is mandatory for business routes when API_KEY is configured.
router = APIRouter(
    tags=["Audit & Alerts"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "/stores/{store_id}/alerts",
    response_model=StoreAlertsOutput,
)
@limiter.limit(config.RATE_LIMIT_ALERTS)
def get_store_alerts(
    request: Request, store_id: str
) -> StoreAlertsOutput:
    """Evaluate inventory rows and rank alerts by operational priority."""
    # The drift status is read from the shared file; no SMTP work happens here.
    model_status = read_model_status()
    candidates = normalize_store_id(store_id)

    try:
        rows = fetch_alert_rows(candidates)
    except DatabaseUnavailableError as exc:
        # Fail loudly: returning only the drift alert would hide inventory
        # anomalies caused by an unavailable database.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable while computing store alerts.",
        ) from exc

    # Pure business rule: level 1 parameter issues, level 2 stock/expiry,
    # level 3 global model drift.
    alerts = build_alerts(rows, model_status)
    return StoreAlertsOutput(
        store_id=store_id,
        model_status=model_status,
        alerts=alerts,
    )
