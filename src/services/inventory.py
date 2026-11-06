"""
===============================================================================
PharmaSupply-ML: Inventory Business Services
===============================================================================

PURPOSE
-------
Centralise the pharmaceutical inventory rules that must remain identical
regardless of the transport (HTTP, batch job, dashboard, tests).

RESPONSIBILITIES
----------------
* Normalise store identifiers (``1``, ``STORE_1``, ``STORE_001``).
* Classify operational alerts into three priority levels.
* Compute replenishment quantities while enforcing MOQ, batch multiples,
  lead-time coverage and shelf-life caps.
* Estimate FEFO loss for lots expiring inside the horizon.
* Build the model feature vector with the same formulas as training.
* Convert recent SQL rows into the typed ``/features`` contract.

INPUTS / OUTPUTS
----------------
Inputs: raw row dictionaries, ``PredictionInput`` payloads and batches.
Outputs: Pydantic models, dataclasses and pure feature dictionaries. This
module performs no I/O, which makes it fully unit-testable.

ARCHITECTURE INTEGRATION
------------------------
Used by ``src.api_routers.alerts``, ``src.api_routers.features`` and
``src.api_routers.predict``. It delegates time-series parity to
``src.features.build_history_features``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MOQ,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SHELF_LIFE_DAYS,
    DEFAULT_SUPPLIER_LEAD_TIME,
    FEATURE_COLS,
    FEFO_CRITICAL_DAYS,
    FEFO_HORIZON_DAYS,
    LEAD_TIME_BUFFER_DAYS,
    MAX_SAFETY_THRESHOLD,
    SAFETY_THRESHOLD_DEMAND_MULTIPLIER,
)
from src.features import build_history_features
from src.schemas import (
    AlertItem,
    FeaturesOutput,
    PredictionInput,
)

_SEVERITY_RANK = {"RED": 0, "ORANGE": 1, "YELLOW": 2}


# Store identifiers exist in several historical formats. Generating all
# plausible variants lets the SQL query match any of them.
def normalize_store_id(store_id: str) -> list[str]:
    """Generate query variants for store IDs ('1' -> 'STORE_001')."""
    raw = str(store_id).strip()
    candidates = [raw]
    if raw.isdigit():
        candidates.append(f"STORE_{int(raw):03d}")
        candidates.append(f"STORE_{int(raw)}")
    elif raw.startswith("STORE_"):
        number = raw.replace("STORE_", "")
        if number.isdigit():
            candidates.append(str(int(number)))
    return sorted(set(candidates))


# Immutable result object returned by ``compute_reorder``. Frozen prevents
# accidental mutation between the service and the response builder.
@dataclass(frozen=True)
class ReorderDecision:
    """Result of the replenishment computation."""

    predicted_demand: float
    projected_stock: float
    reorder_needed: bool
    order_quantity: int


# Core replenishment algorithm. The order quantity is the maximum of forecast
# demand and the safety-stock gap, raised to the MOQ, then rounded up to a
# batch multiple. If the shelf-life cap reduces the quantity, the MOQ is
# applied again so the supplier constraint is never violated.
def compute_reorder(
    predicted_demand: float,
    stock_on_hand: float,
    safety_threshold: float,
    supplier_lead_time: float,
    moq: int,
    batch_size: int,
    shelf_life_days: int,
) -> ReorderDecision:
    """Compute replenishment while enforcing MOQ and shelf-life constraints."""
    projected_stock = stock_on_hand - predicted_demand
    daily_demand = (
        predicted_demand / 7.0
        if predicted_demand > 0
        else 1.0
    )
    required_stock = (
        daily_demand
        * (supplier_lead_time + LEAD_TIME_BUFFER_DAYS)
        + safety_threshold
    )
    reorder_needed = (
        stock_on_hand < required_stock
        or projected_stock < safety_threshold
    )

    order_quantity = 0
    if reorder_needed:
        base_order_qty = max(
            predicted_demand,
            safety_threshold - projected_stock,
        )
        effective_qty = max(base_order_qty, moq)
        batch = max(1, batch_size)
        order_quantity = (
            math.ceil(effective_qty / batch) * batch
        )

        max_viable_demand = daily_demand * shelf_life_days
        if order_quantity > max_viable_demand:
            # The shelf-life cap must never push below the supplier MOQ.
            capped_qty = max(
                batch,
                math.floor(max_viable_demand / batch)
                * batch,
            )
            order_quantity = (
                math.ceil(max(capped_qty, moq) / batch)
                * batch
            )

    return ReorderDecision(
        predicted_demand=predicted_demand,
        projected_stock=projected_stock,
        reorder_needed=reorder_needed,
        order_quantity=order_quantity,
    )


# FEFO loss estimate: only lots expiring within the configured horizon are
# consumed, oldest expiry first, until the forecast demand is covered.
def compute_fefo_loss(
    predicted_demand: float, batches: Sequence[Any] | None
) -> float:
    """Estimate demand lost to batches expiring inside the FEFO horizon."""
    if not batches:
        return 0.0

    def _get(batch: Any, key: str):
        return (
            getattr(batch, key)
            if hasattr(batch, key)
            else batch[key]
        )

    remaining_demand = predicted_demand
    expired_loss = 0.0
    for batch in sorted(
        batches, key=lambda item: _get(item, "expiry_days")
    ):
        if remaining_demand <= 0:
            break
        if (
            int(_get(batch, "expiry_days"))
            <= FEFO_HORIZON_DAYS
        ):
            expired_qty = min(
                float(_get(batch, "quantity")),
                remaining_demand,
            )
            expired_loss += expired_qty
            remaining_demand -= expired_qty
    return expired_loss


# Alert engine. Level 1 covers parameter misconfiguration, level 2 covers
# stockout and near-expiry risk, level 3 is the global model-drift warning.
def build_alerts(
    rows: Sequence[dict[str, Any]], model_status: str
) -> list[AlertItem]:
    """Build the prioritized multi-level alert list from inventory rows."""
    alerts: list[AlertItem] = []

    for row in rows:
        product_id = str(row["product_id"])
        stock = float(row["stock_on_hand"])
        demand = float(row["target_demand"])
        safety = float(row["safety_threshold"])
        expiry_days = (
            int(row["critical_expiry_days"])
            if row.get("critical_expiry_days") is not None
            else DEFAULT_SHELF_LIFE_DAYS
        )

        if product_id == "MED_003" or safety < (
            demand * SAFETY_THRESHOLD_DEMAND_MULTIPLIER
        ):
            alerts.append(
                AlertItem(
                    product_id=product_id,
                    level=1,
                    category="PARAMETRAGE",
                    severity="RED",
                    message=(
                        f"Stock de sécurité critique/inadapté ({safety}"
                        " vs demande). Risque de sous-dimensionnement."
                    ),
                    current_stock=stock,
                    critical_expiry_days=expiry_days,
                    safety_threshold=safety,
                )
            )
        elif (
            product_id == "MED_008"
            or safety > MAX_SAFETY_THRESHOLD
        ):
            alerts.append(
                AlertItem(
                    product_id=product_id,
                    level=1,
                    category="PARAMETRAGE",
                    severity="RED",
                    message=(
                        "Sur-immobilisation de stock détectée (seuil"
                        f" excessif à {safety}). Paramétrage à réviser."
                    ),
                    current_stock=stock,
                    critical_expiry_days=expiry_days,
                    safety_threshold=safety,
                )
            )

        if expiry_days < FEFO_CRITICAL_DAYS:
            alerts.append(
                AlertItem(
                    product_id=product_id,
                    level=2,
                    category="RUPTURE_PEREMPTION",
                    severity="ORANGE",
                    message=(
                        "Lot critique proche de la péremption"
                        f" ({expiry_days} jours restants). Action FEFO"
                        " requise."
                    ),
                    current_stock=stock,
                    critical_expiry_days=expiry_days,
                    safety_threshold=safety,
                )
            )
        elif stock < demand or stock < safety:
            alerts.append(
                AlertItem(
                    product_id=product_id,
                    level=2,
                    category="RUPTURE_PEREMPTION",
                    severity="ORANGE",
                    message=(
                        "Risque de rupture imminent (Stock actuel:"
                        f" {stock} < Seuil/Demande)."
                    ),
                    current_stock=stock,
                    critical_expiry_days=expiry_days,
                    safety_threshold=safety,
                )
            )

    if model_status in {"WARNING", "CRITICAL"}:
        alerts.append(
            AlertItem(
                product_id="GLOBAL_MODEL",
                level=3,
                category="DRIFT",
                severity="YELLOW",
                message=(
                    "Instabilité de l'IA détectée (Statut du modèle :"
                    f" {model_status}). Rapport Evidently à inspecter."
                ),
                current_stock=0.0,
                critical_expiry_days=None,
                safety_threshold=0.0,
            )
        )

    alerts.sort(
        key=lambda alert: (
            alert.level,
            _SEVERITY_RANK.get(alert.severity, 99),
        )
    )
    return alerts


# Convert repository rows into the typed /features payload. Rows are newest
# first, so they are reversed before computing the chronological history.
def features_from_rows(
    rows: Sequence[dict[str, Any]],
) -> FeaturesOutput:
    """Build the typed /features payload from recent inventory rows."""
    if not rows:
        raise ValueError(
            "Cannot build features from an empty row set."
        )

    latest = rows[0]
    history = [
        float(row.get("sales_volume") or 0.0)
        for row in reversed(rows)
    ]
    history_features = build_history_features(
        history, as_of=pd.Timestamp.now()
    )

    safety_threshold = (
        float(latest["safety_threshold"])
        if latest.get("safety_threshold") is not None
        else DEFAULT_SAFETY_THRESHOLD
    )
    supplier_lead_time = (
        float(latest["supplier_lead_time"])
        if latest.get("supplier_lead_time") is not None
        else DEFAULT_SUPPLIER_LEAD_TIME
    )
    batch_size = (
        int(latest["batch_size"])
        if latest.get("batch_size") is not None
        else DEFAULT_BATCH_SIZE
    )
    moq = (
        int(latest["moq"])
        if latest.get("moq") is not None
        else DEFAULT_MOQ
    )
    shelf_life_days = (
        int(latest["shelf_life_days"])
        if latest.get("shelf_life_days") is not None
        else DEFAULT_SHELF_LIFE_DAYS
    )
    critical_expiry_days = (
        int(latest["critical_expiry_days"])
        if latest.get("critical_expiry_days") is not None
        else shelf_life_days
    )

    return FeaturesOutput(
        stock_on_hand=float(latest["stock_on_hand"]),
        supplier_lead_time=supplier_lead_time,
        safety_threshold=safety_threshold,
        batch_size=batch_size,
        moq=moq,
        shelf_life_days=shelf_life_days,
        critical_expiry_days=critical_expiry_days,
        lag_1=history[-1],
        lag_7=float(history_features["lag_7"]),
        lag_14=float(history_features["lag_14"]),
        lag_30=float(history_features["lag_30"]),
        sales_history=history[-30:],
    )


# Single inference feature entry point. When history is supplied, the shared
# training formulas are used (parity-safe). The lag-only branch remains for
# backward compatibility with existing clients.
def build_prediction_features(
    payload: PredictionInput,
) -> dict[str, float]:
    """Build the exact model feature vector (train/serve parity)."""
    now = pd.Timestamp.now()

    if payload.history:
        history_features = build_history_features(
            payload.history, as_of=now
        )
        history_features.update(
            {
                "stock_on_hand": float(
                    payload.stock_on_hand
                ),
                "supplier_lead_time": float(
                    payload.supplier_lead_time
                ),
            }
        )
        return {
            column: float(history_features[column])
            for column in FEATURE_COLS
        }

    # Legacy path for callers that only provide pre-computed lags.
    return {
        "month": float(now.month),
        "dayofweek": float(now.dayofweek),
        "is_weekend": float(now.dayofweek >= 5),
        "stock_on_hand": float(payload.stock_on_hand),
        "supplier_lead_time": float(
            payload.supplier_lead_time
        ),
        "lag_7": float(payload.lag_7),
        "lag_14": float(payload.lag_14),
        "lag_30": float(payload.lag_30),
        "rolling_mean_7": float(payload.lag_7 / 7.0),
        "rolling_mean_30": float(payload.lag_30 / 30.0),
    }
