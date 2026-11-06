"""
Health, liveness and readiness endpoints.

RESPONSIBILITIES
----------------
* ``/model/health`` exposes the model drift status and its operator-facing
  message. It is strictly side-effect free: notifications are handled by the
  background watcher, never by a probe.
* ``/livez`` proves that the process can answer HTTP requests (liveness).
* ``/readyz`` proves that the model is loaded and PostgreSQL is reachable
  (readiness). Container orchestrators and ``run_pipeline.py`` use it before
  sending traffic or running smoke tests.

INPUTS / OUTPUTS
----------------
Inputs: the shared status file and a trivial database query.
Outputs: JSON payloads, or ``503`` from ``/readyz`` when a dependency is down.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from src.config import HEALTH_MESSAGES
from src.repositories.database import check_database_health
from src.schemas import (
    LivezOutput,
    ModelHealthOutput,
    ReadyzOutput,
)
from src.services.model_status import read_model_status

# Health routes are intentionally unauthenticated so infrastructure probes and
# load balancers can call them without secrets.
router = APIRouter(tags=["Audit & Alerts"])


@router.get(
    "/model/health", response_model=ModelHealthOutput
)
def get_model_health() -> ModelHealthOutput:
    """Expose the model health status without triggering notifications.

    Unknown or unreadable statuses are mapped to the fail-safe ``UNKNOWN``
    message rather than being presented as ``STABLE``.
    """
    current_status = read_model_status()
    return ModelHealthOutput(
        status=current_status,
        message=HEALTH_MESSAGES.get(
            current_status, HEALTH_MESSAGES["UNKNOWN"]
        ),
    )


@router.get("/livez", response_model=LivezOutput)
def livez() -> LivezOutput:
    """Liveness probe: the process is able to answer requests."""
    return LivezOutput(status="alive")


@router.get("/readyz", response_model=ReadyzOutput)
def readyz(request: Request) -> ReadyzOutput:
    """Readiness probe: model loaded and database reachable.

    Returns ``503`` with a structured detail when either dependency is not
    ready. This endpoint is safe to poll: it performs no writes and sends no
    e-mail.
    """
    model_loaded = (
        getattr(request.app.state, "model", None)
        is not None
    )
    database_ready = check_database_health()

    if not (model_loaded and database_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "model_loaded": model_loaded,
                "database_ready": database_ready,
            },
        )

    return ReadyzOutput(
        status="ready",
        model_loaded=True,
        database_ready=True,
    )
