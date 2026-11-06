"""
===============================================================================
PharmaSupply-ML: Observability & Production Drift Inputs
===============================================================================

PURPOSE
-------
Make the serving layer observable in production and collect the aggregated
feature data required by production drift monitoring.

RESPONSIBILITIES
----------------
* Instrument every HTTP request with a request id, JSON access log, counter
  and latency histogram.
* Expose Prometheus metrics through ``GET /metrics``.
* Record one non-identifying JSON line per prediction (features + output) into
  ``reports/prediction_log.jsonl``; this file is the input of
  ``src.monitoring.run_production_drift_monitoring``.
* Provide safe readers for the prediction log used by tests and batch jobs.

INPUTS / OUTPUTS
----------------
* Inputs: FastAPI requests/responses and prediction feature dictionaries.
* Outputs: structured log lines, Prometheus metrics, JSONL records on disk.

ARCHITECTURE INTEGRATION
------------------------
``setup_observability`` is called by ``src.api.create_app``. ``record_prediction``
is scheduled as a FastAPI background task by the predict router, so file I/O
never blocks inference.

OPERATIONAL NOTES
-----------------
No personal or customer data is logged. Predicted demand and model features
are sufficient to detect distribution shifts. Log rotation/retention must be
handled by the platform.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
)
from starlette.middleware.base import BaseHTTPMiddleware

import src.config as config
from src.config import FEATURE_COLS

logger = logging.getLogger("pharmasupply.observability")

# ---------------------------------------------------------------------------
# Prometheus instruments. Counters and histograms are registered once at import
# time; labels stay low-cardinality by using the route template, not raw URLs.
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "pharma_api_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "pharma_api_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)
PREDICTION_DEMAND = Histogram(
    "pharma_api_predicted_demand",
    "Predicted 7-day demand",
    buckets=(
        10,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
        10000,
    ),
)
REORDER_TOTAL = Counter(
    "pharma_api_reorders_total",
    "Recommendations requiring a reorder",
)

# Mounted under /metrics. A single ASGI app instance is reused across app
# factories to avoid duplicate metric registration in the same process.
_METRICS_APP = make_asgi_app()
_LOG_LOCK = threading.Lock()


# Middleware order matters: it wraps the request to measure the full handler
# duration and to attach the request id to the response headers.
class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach a request id, emit JSON access logs and record metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = (
            request.headers.get("X-Request-ID")
            or uuid.uuid4().hex
        )
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (
                time.perf_counter() - started
            ) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        route = request.scope.get("route")
        path_label = getattr(
            route, "path", request.url.path
        )
        REQUEST_COUNT.labels(
            request.method,
            path_label,
            str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            request.method, path_label
        ).observe(duration_ms / 1000)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": path_label,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


# Called from a FastAPI BackgroundTask. The lock makes appends safe when the
# API runs with multiple threads; OSError is swallowed because logging must
# never break inference.
def record_prediction(
    features: dict[str, float],
    predicted_demand: float,
    reorder_needed: bool,
    store_id: Any,
    product_id: Any,
) -> None:
    """Append an aggregated, non-identifying prediction record to JSONL.

    Only model features and the prediction are stored; no customer data is
    collected. This log is the input for production drift monitoring.
    """
    PREDICTION_DEMAND.observe(predicted_demand)
    if reorder_needed:
        REORDER_TOTAL.inc()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "store_id": str(store_id),
        "product_id": str(product_id),
        "predicted_demand": float(predicted_demand),
        "reorder_needed": bool(reorder_needed),
    }
    record.update(
        {
            column: float(features[column])
            for column in FEATURE_COLS
            if column in features
        }
    )

    path = Path(config.PREDICTION_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except (
        OSError
    ) as exc:  # logging must never break inference
        logger.warning(
            "Unable to persist prediction log: %s", exc
        )


# Tolerant reader: malformed lines are skipped so a partially written file
# cannot break a drift batch.
def read_prediction_log(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read the JSONL prediction log, ignoring malformed lines."""
    log_path = Path(path or config.PREDICTION_LOG_PATH)
    if not log_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed prediction log line."
                )
    return records


# Application wiring: middleware + Prometheus endpoint.
def setup_observability(app: FastAPI) -> None:
    """Register middleware and expose the Prometheus endpoint."""
    app.add_middleware(ObservabilityMiddleware)
    app.mount("/metrics", _METRICS_APP)
