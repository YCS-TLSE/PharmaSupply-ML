"""
===============================================================================
PharmaSupply-ML: FastAPI Application Assembly
===============================================================================

PURPOSE
-------
Build the FastAPI application that serves pharmaceutical demand forecasts,
inventory alerts and product features. This module is intentionally thin: it
only wires routers, security, observability and the application lifespan.

RESPONSIBILITIES
----------------
* Load the serialised XGBoost artifact once at startup and expose it through
  ``app.state.model``.
* Start the transition-based drift-alert watcher and cancel it cleanly on
  shutdown.
* Configure structured logging, API security (key/CORS/rate limiting) and
  Prometheus/HTTP observability.
* Mount every router under a single application object.

INPUTS / OUTPUTS
----------------
* Inputs: ``MODEL_PATH`` and other settings from ``src.config``.
* Outputs: the global ``app`` object consumed by ``uvicorn src.api:app``.
  HTTP request/response contracts are defined in ``src.schemas``.

ARCHITECTURE INTEGRATION
------------------------
``src.api`` is the composition root of the serving layer:

    api_routers -> services -> repositories -> PostgreSQL
                       |
                       +-> features/train (model contract)
                       +-> observability (metrics + prediction log)
                       +-> model_status (drift notifications)

Keeping the composition root separate from the routers allows the HTTP layer
to be tested with dependency overrides while business rules stay framework
independent.

OPERATIONAL NOTES
-----------------
The model is loaded per process. Running multiple uvicorn workers therefore
loads one copy of the artifact per worker, which must be accounted for in the
container memory budget.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

import joblib
from fastapi import FastAPI

from src.api_routers import (
    alerts,
    data,
    features,
    health,
    predict,
)
from src.config import MODEL_PATH
from src.logging_config import configure_logging
from src.observability import setup_observability
from src.security import configure_security
from src.services.model_status import watch_model_status

logger = logging.getLogger("pharmasupply.api")

# Module-level handle kept for backward compatibility with code that imported
# ``src.api.loaded_model``. Routers use ``app.state.model`` instead, which is
# the source of truth and can be overridden in tests.
loaded_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, start the drift watcher, and clean up on shutdown.

    The artifact is loaded once to avoid per-request disk I/O. The watcher is
    started as an asyncio task so SMTP notifications never run in the request
    path. The ``finally`` block guarantees task cancellation and state cleanup
    when the server stops or reloads.
    """
    global loaded_model

    # ------------------------------------------------------------------
    # Startup: load the model artifact into process memory.
    # ------------------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                "Model successfully loaded from: %s",
                MODEL_PATH,
            )
        except Exception as exc:
            logger.error(
                "Error loading model artifact: %s", exc
            )
            loaded_model = None
    else:
        logger.warning(
            "Model artifact not found at '%s'.", MODEL_PATH
        )

    app.state.model = loaded_model

    # ------------------------------------------------------------------
    # Background watcher: reads model_status.txt and sends transition alerts.
    # ------------------------------------------------------------------
    status_task = asyncio.create_task(watch_model_status())
    try:
        yield
    finally:
        # Shutdown: cancel the watcher and release the model reference.
        status_task.cancel()
        with suppress(asyncio.CancelledError):
            await status_task
        app.state.model = None
        loaded_model = None


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    # JSON logging must be configured before the first log record is emitted.
    configure_logging()

    application = FastAPI(
        title="PharmaSupply-ML Demand Forecasting API",
        description=(
            "Decision-making MLOps API for pharmacy inventory replenishment."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Cross-cutting concerns: authentication/CORS/rate limiting first, then
    # request instrumentation and the /metrics endpoint.
    configure_security(application)
    setup_observability(application)

    # Business routers. Order has no functional impact because paths are
    # disjoint, but grouping keeps the OpenAPI document readable.
    application.include_router(health.router)
    application.include_router(data.router)
    application.include_router(alerts.router)
    application.include_router(features.router)
    application.include_router(predict.router)

    logger.info(
        "Application assembled with %d routes.",
        len(application.routes),
    )
    return application


# Global ASGI application used by ``uvicorn src.api:app``. Tests import this
# object and override dependencies when they need deterministic behaviour.
app = create_app()
