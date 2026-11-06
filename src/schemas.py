"""
===============================================================================
PharmaSupply-ML: Pydantic API Contracts
===============================================================================

PURPOSE
-------
Define the typed request and response schemas exposed by the FastAPI service.
These models are the boundary between untrusted HTTP input and trusted
business logic: FastAPI validates them before any endpoint body executes.

RESPONSIBILITIES
----------------
* Validate inference payloads (identifiers, stock, lags, history, batches).
* Describe response payloads for OpenAPI documentation and contract tests.
* Enforce numeric bounds (``ge=0``, ``ge=1``, ``min_length``).
* Carry the model version so clients can detect artifact changes.

INPUTS / OUTPUTS
----------------
* Inputs: JSON objects received by ``POST /predict`` and query/path params.
* Outputs: serialisable models returned by the routers. Invalid input produces
  a FastAPI ``422 Unprocessable Entity`` before the service layer is reached.

ARCHITECTURE INTEGRATION
------------------------
Imported by ``src.api_routers`` and ``src.services.inventory``. The optional
``history`` field activates the shared train/serve feature builder; when it is
absent, the legacy lag-based path remains available for compatibility.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from src.config import MODEL_VERSION


# ---------------------------------------------------------------------------
# FEFO batch input: one physical lot with a remaining shelf life in days.
# ---------------------------------------------------------------------------
class BatchItem(BaseModel):
    batch_id: str = Field(
        ..., json_schema_extra={"example": "BATCH_001"}
    )
    quantity: float = Field(
        ..., ge=0, json_schema_extra={"example": 100.0}
    )
    expiry_days: int = Field(
        ..., ge=0, json_schema_extra={"example": 45}
    )


# ---------------------------------------------------------------------------
# Inference payload. ``history`` is the parity-safe path: when provided, the
# API rebuilds features with the exact training formulas and ignores lag_*.
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    store_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "1"}
    )
    product_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "MED_001"}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 100.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    safety_threshold: float = Field(
        ..., ge=0, json_schema_extra={"example": 150.0}
    )
    batch_size: int = Field(
        default=1, ge=1, json_schema_extra={"example": 50}
    )
    moq: int = Field(
        default=1, ge=1, json_schema_extra={"example": 100}
    )
    shelf_life_days: int = Field(
        default=365,
        ge=1,
        json_schema_extra={"example": 365},
    )
    batches: Optional[List[BatchItem]] = Field(
        default=None,
        description="Optional list of active batches for FEFO tracking",
    )
    lag_1: Optional[float] = Field(default=0.0, ge=0)
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 630.0}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 1260.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 2700.0}
    )
    history: Optional[List[float]] = Field(
        default=None,
        min_length=30,
        description=(
            "Optional chronological daily sales history (at least 30 "
            "values, oldest first). When provided, the shared feature "
            "builder is used and the lag_* fields are ignored."
        ),
    )


# ---------------------------------------------------------------------------
# Inference result: demand forecast plus constraint-compliant replenishment.
# ---------------------------------------------------------------------------
class PredictionOutput(BaseModel):
    store_id: Union[int, str]
    product_id: Union[int, str]
    predicted_demand: float
    projected_stock: float
    reorder_needed: bool
    order_quantity: int
    fefo_expired_loss: float = 0.0
    model_status: str
    model_version: str = MODEL_VERSION


# ---------------------------------------------------------------------------
# Reference-data responses used by the dashboard selectors.
# ---------------------------------------------------------------------------
class StoresOutput(BaseModel):
    stores: List[Union[int, str]]


class ProductsOutput(BaseModel):
    products: List[Union[int, str]]


# ---------------------------------------------------------------------------
# Typed /features contract. ``sales_history`` is chronological (oldest first)
# and is forwarded by the dashboard to /predict for train/serve parity.
# ---------------------------------------------------------------------------
class FeaturesOutput(BaseModel):
    """Typed contract for ``GET /features/{store_id}/{product_id}``."""

    stock_on_hand: float
    supplier_lead_time: float
    safety_threshold: float
    batch_size: int
    moq: int
    shelf_life_days: int
    critical_expiry_days: int
    lag_1: float
    lag_7: float
    lag_14: float
    lag_30: float
    sales_history: List[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Operational alert (level 1: parameter, level 2: stockout/expiry, level 3: drift).
# ---------------------------------------------------------------------------
class AlertItem(BaseModel):
    product_id: str
    level: int = Field(
        ...,
        description="1: Paramétrage, 2: Rupture/Péremption, 3: Drift",
    )
    category: str = Field(
        ...,
        description="PARAMETRAGE, RUPTURE_PEREMPTION ou DRIFT",
    )
    severity: str = Field(
        ..., description="RED, ORANGE ou YELLOW"
    )
    message: str
    current_stock: float
    critical_expiry_days: Optional[int] = None
    safety_threshold: float


class StoreAlertsOutput(BaseModel):
    store_id: Union[int, str]
    model_status: str
    alerts: List[AlertItem]


# ---------------------------------------------------------------------------
# Health and probe payloads. ``UNKNOWN`` is a first-class fail-safe state.
# ---------------------------------------------------------------------------
class ModelHealthOutput(BaseModel):
    status: str
    message: str


class LivezOutput(BaseModel):
    status: str


class ReadyzOutput(BaseModel):
    status: str
    model_loaded: bool
    database_ready: bool
