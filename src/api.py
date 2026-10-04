"""
===============================================================================
PharmaSupply-ML: Demand Forecasting API Service
===============================================================================

Description:
------------
This module implements a production-ready REST API using FastAPI for serving the
trained XGBoost pharmaceutical demand forecasting model. It handles input
data validation via Pydantic schemas, dynamically loads the serialized model
artifact (`.joblib`) during startup using FastAPI lifespan context, and exposes
endpoints for health monitoring and batch inference.

Author: YCS127
Project: PharmaSupply-ML
===============================================================================
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmasupply-api")

# =============================================================================
# 1. CONFIGURATION & GLOBALS
# =============================================================================
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)
loaded_model = None

FEATURE_COLS = [
    "month",
    "dayofweek",
    "is_weekend",
    "stock_on_hand",
    "supplier_lead_time",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]


# =============================================================================
# 2. DATA VALIDATION SCHEMAS (Pydantic v2)
# =============================================================================
class PredictionInput(BaseModel):
    store_id: int = Field(
        ..., json_schema_extra={"example": 101}
    )
    product_id: int = Field(
        ..., json_schema_extra={"example": 2045}
    )
    month: int = Field(
        ..., ge=1, le=12, json_schema_extra={"example": 10}
    )
    dayofweek: int = Field(
        ..., ge=0, le=6, json_schema_extra={"example": 2}
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, json_schema_extra={"example": 0}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 500.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 120.5}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 115.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 110.0}
    )
    rolling_mean_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 118.2}
    )
    rolling_mean_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 112.4}
    )


class PredictionOutput(BaseModel):
    store_id: int
    product_id: int
    predicted_demand: float
    model_version: str = "1.0.0"


class HealthCheckOutput(BaseModel):
    status: str
    model_loaded: bool


# =============================================================================
# 3. LIFESPAN MANAGEMENT
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model

    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from: {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(
                f"❌ Error loading model from {MODEL_PATH}: {e}"
            )
            loaded_model = None
    else:
        logger.warning(
            f"⚠️ Model file not found at '{MODEL_PATH}'. /predict will return HTTP 503."
        )

    yield

    logger.info(
        "🧹 Unloading ML model and cleaning up resources..."
    )
    loaded_model = None


# =============================================================================
# 4. FASTAPI APP INITIALIZATION
# =============================================================================
app = FastAPI(
    title="PharmaSupply-ML Demand Forecasting API",
    description="Production-grade API serving XGBoost predictions for pharmaceutical supply chain planning.",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# 5. API ROUTE ENDPOINTS
# =============================================================================
@app.get(
    "/health",
    response_model=HealthCheckOutput,
    tags=["Health"],
)
async def health_check():
    return HealthCheckOutput(
        status="healthy",
        model_loaded=loaded_model is not None,
    )


@app.post(
    "/predict",
    response_model=List[PredictionOutput],
    tags=["Inference"],
)
async def predict(payload: List[PredictionInput]):
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Model is not loaded.",
        )

    try:
        df = pd.DataFrame(
            [item.model_dump() for item in payload]
        )
        X = df[FEATURE_COLS]

        # Inférence et post-traitement métiers (pas de demande négative)
        raw_preds = loaded_model.predict(X)
        clipped_preds = np.maximum(0.0, raw_preds)

        results = [
            PredictionOutput(
                store_id=item.store_id,
                product_id=item.product_id,
                predicted_demand=float(pred),
                model_version="1.0.0",
            )
            for item, pred in zip(payload, clipped_preds)
        ]
        return results

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference execution failed: {str(e)}",
        )
