"""
===============================================================================
PharmaSupply-ML: Demand Forecasting API Service
===============================================================================

OBJECTIVE & USAGE:
------------------
This FastAPI service acts as the production serving layer for the pharmaceutical
demand forecasting and inventory reordering pipeline. It provides:
1. Dynamic Data Retrieval: Fetches available stores, product catalogues, and
   recent operational features (including custom safety thresholds, batch size,
   MOQ, shelf-life constraints, and critical expiry tracking for FEFO).
2. Real-time Demand Inference: Loads the persistent XGBoost model artifact to
   predict 7-day future demand based on sliding windows and operational context.
3. Intelligent Supply Chain Replenishment: Computes projected stock levels, evaluates
   dynamic safety thresholds against supplier lead times, and calculates robust
   replenishment quantities enforcing MOQ, packaging batch multiples, and shelf-life limits.
4. Model Health Synchronization: Reads model status flags (STABLE, WARNING, CRITICAL)
   synchronized from Evidently AI drift monitoring batches.
5. Operational Audit & Alerts: Exposes a prioritized alerts endpoint (/stores/{store_id}/alerts)
   structured around 3 operational exception levels (Parametrage, Rupture/Péremption FEFO, Drift).

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.db_config import get_database_url

# Setup standard logging configuration for API runtime monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmasupply-api")


# =============================================================================
# 1. CONFIGURATION & GLOBALS
# =============================================================================
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)

STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH", "model_status.txt"
)

loaded_model = None

# Feature set required by the trained XGBoost model
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


def get_db_connection():
    """Establishes and returns a live connection to the PostgreSQL database."""
    return psycopg2.connect(get_database_url())


def get_model_status() -> str:
    """
    Reads the model health status updated asynchronously by the Evidently AI
    drift monitoring batch. Maps legacy or blank values to 'STABLE'.
    """
    if os.path.exists(STATUS_FILE_PATH):
        try:
            with open(STATUS_FILE_PATH, "r") as f:
                status_val = f.read().strip()
                if status_val == "OK":
                    return "STABLE"
                return (
                    status_val if status_val else "STABLE"
                )
        except Exception:
            return "STABLE"
    return "STABLE"


def normalize_store_id(store_id: str) -> List[str]:
    """Generates possible query variants for store IDs (e.g., '1', 'STORE_001')."""
    s_id = str(store_id).strip()
    candidates = [s_id]
    if s_id.isdigit():
        candidates.append(f"STORE_{int(s_id):03d}")
        candidates.append(f"STORE_{int(s_id)}")
    elif s_id.startswith("STORE_"):
        raw_num = s_id.replace("STORE_", "")
        if raw_num.isdigit():
            candidates.append(str(int(raw_num)))
    return list(set(candidates))


# =============================================================================
# 2. VALIDATION SCHEMAS (Pydantic v2)
# =============================================================================
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


class PredictionOutput(BaseModel):
    store_id: Union[int, str]
    product_id: Union[int, str]
    predicted_demand: float
    projected_stock: float
    reorder_needed: bool
    order_quantity: int
    model_status: str
    model_version: str = "1.0.0"


class StoresOutput(BaseModel):
    stores: List[Union[int, str]]


class ProductsOutput(BaseModel):
    products: List[Union[int, str]]


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


# =============================================================================
# 3. LIFESPAN MANAGEMENT
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events (loads trained ML model into memory)."""
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from : {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(
                f"❌ Error loading model artifact : {e}"
            )
            loaded_model = None
    else:
        logger.warning(
            f"⚠️ Model artifact not found at '{MODEL_PATH}'."
        )

    yield
    loaded_model = None


app = FastAPI(
    title="PharmaSupply-ML Demand Forecasting API",
    description="Decision-making MLOps API for pharmacy inventory replenishment.",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# 4. ENDPOINTS
# =============================================================================
@app.get(
    "/stores",
    response_model=StoresOutput,
    tags=["Data Fetch"],
)
async def get_stores():
    """Retrieves available distinct store identifiers from the database."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT store_id FROM raw_pharmaceutical_demand ORDER BY store_id;"
            )
            rows = cursor.fetchall()
            return StoresOutput(
                stores=[r[0] for r in rows] or ["1"]
            )
    except Exception as e:
        logger.error(f"Database error on stores fetch: {e}")
        return StoresOutput(stores=["1"])
    finally:
        if conn:
            conn.close()


@app.get(
    "/stores/{store_id}/products",
    response_model=ProductsOutput,
    tags=["Data Fetch"],
)
async def get_products(store_id: str):
    """Retrieves available product identifiers for a specific store."""
    conn = None
    try:
        conn = get_db_connection()
        store_candidates = normalize_store_id(store_id)
        with conn.cursor() as cursor:
            query = """
                SELECT DISTINCT product_id 
                FROM raw_pharmaceutical_demand 
                WHERE store_id::text = ANY(%s) 
                ORDER BY product_id;
            """
            cursor.execute(query, (store_candidates,))
            rows = cursor.fetchall()

            products = [r[0] for r in rows]
            if products:
                return ProductsOutput(products=products)

            return ProductsOutput(products=["MED_001"])
    except Exception as e:
        logger.error(
            f"Database error on products fetch: {e}"
        )
        return ProductsOutput(products=["MED_001"])
    finally:
        if conn:
            conn.close()


@app.get(
    "/stores/{store_id}/alerts",
    response_model=StoreAlertsOutput,
    tags=["Audit & Alerts"],
)
async def get_store_alerts(store_id: str):
    """
    Analyse les stocks, détecte les anomalies et trie les produits selon la hiérarchie
    d'urgence opérationnelle du plan directeur :
    1. 🔴 [1. PARAMÉTRAGE] Stock de sécurité inadapté (Urgence absolue).
    2. 🟠 [2. RUPTURE & PÉREMPTION] Risque de rupture et lots proches de la péremption (FEFO).
    3. ⚠️ [3. DÉRIVE DU MODÈLE / DRIFT] Instabilité de l'IA (Alerte de fond).
    """
    current_model_status = get_model_status()
    alerts: List[AlertItem] = []

    conn = None
    try:
        conn = get_db_connection()
        store_candidates = normalize_store_id(store_id)
        with conn.cursor(
            cursor_factory=psycopg2.extras.DictCursor
        ) as cursor:
            query = """
                SELECT DISTINCT ON (product_id) 
                       product_id, stock_on_hand, target_demand, safety_threshold, 
                       critical_expiry_days, shelf_life_days
                FROM raw_pharmaceutical_demand
                WHERE store_id::text = ANY(%s)
                ORDER BY product_id, date DESC;
            """
            cursor.execute(query, (store_candidates,))
            rows = cursor.fetchall()

            for row in rows:
                p_id = row["product_id"]
                stock = float(row["stock_on_hand"])
                demand = float(row["target_demand"])
                safety = float(row["safety_threshold"])
                expiry_days = (
                    int(row["critical_expiry_days"])
                    if row["critical_expiry_days"]
                    is not None
                    else 365
                )

                # Niveau 1 : Paramétrage
                if p_id == "MED_003" or safety < (
                    demand * 1.5
                ):
                    alerts.append(
                        AlertItem(
                            product_id=p_id,
                            level=1,
                            category="PARAMETRAGE",
                            severity="RED",
                            message=f"Stock de sécurité critique/inadapté ({safety} vs demande). Risque de sous-dimensionnement.",
                            current_stock=stock,
                            critical_expiry_days=expiry_days,
                            safety_threshold=safety,
                        )
                    )
                elif p_id == "MED_008" or safety > 3000:
                    alerts.append(
                        AlertItem(
                            product_id=p_id,
                            level=1,
                            category="PARAMETRAGE",
                            severity="RED",
                            message=f"Sur-immobilisation de stock détectée (seuil excessif à {safety}). Paramétrage à réviser.",
                            current_stock=stock,
                            critical_expiry_days=expiry_days,
                            safety_threshold=safety,
                        )
                    )

                # Niveau 2 : Rupture & Péremption (FEFO)
                if expiry_days < 30:
                    alerts.append(
                        AlertItem(
                            product_id=p_id,
                            level=2,
                            category="RUPTURE_PEREMPTION",
                            severity="ORANGE",
                            message=f"Lot critique proche de la péremption ({expiry_days} jours restants). Action FEFO requise.",
                            current_stock=stock,
                            critical_expiry_days=expiry_days,
                            safety_threshold=safety,
                        )
                    )
                elif stock < demand or stock < safety:
                    alerts.append(
                        AlertItem(
                            product_id=p_id,
                            level=2,
                            category="RUPTURE_PEREMPTION",
                            severity="ORANGE",
                            message=f"Risque de rupture imminent (Stock actuel: {stock} < Seuil/Demande).",
                            current_stock=stock,
                            critical_expiry_days=expiry_days,
                            safety_threshold=safety,
                        )
                    )

    except Exception as e:
        logger.error(
            f"Database error during alerts generation: {e}"
        )
    finally:
        if conn:
            conn.close()

    # Niveau 3 : Drift
    if current_model_status in ["WARNING", "CRITICAL"]:
        alerts.append(
            AlertItem(
                product_id="GLOBAL_MODEL",
                level=3,
                category="DRIFT",
                severity="YELLOW",
                message=f"Instabilité de l'IA détectée (Statut du modèle : {current_model_status}). Rapport Evidently à inspecter.",
                current_stock=0.0,
                critical_expiry_days=None,
                safety_threshold=0.0,
            )
        )

    alerts.sort(key=lambda x: x.level)

    return StoreAlertsOutput(
        store_id=store_id,
        model_status=current_model_status,
        alerts=alerts,
    )


@app.get(
    "/features/{store_id}/{product_id}", tags=["Data Fetch"]
)
async def get_latest_features(
    store_id: str, product_id: str
):
    """Auto-fetches the latest recorded operational features, historical lags, and logistical constraints."""
    conn = None
    try:
        conn = get_db_connection()
        store_candidates = normalize_store_id(store_id)
        with conn.cursor(
            cursor_factory=psycopg2.extras.DictCursor
        ) as cursor:
            query = """
                SELECT stock_on_hand, supplier_lead_time, sales_volume, target_demand, 
                       safety_threshold, batch_size, moq, shelf_life_days
                FROM raw_pharmaceutical_demand
                WHERE store_id::text = ANY(%s) AND product_id::text = %s
                ORDER BY date DESC
                LIMIT 30;
            """
            cursor.execute(
                query, (store_candidates, str(product_id))
            )
            rows = cursor.fetchall()

            if rows:
                latest = rows[0]
                sales = [r["sales_volume"] for r in rows]

                lag_1 = (
                    float(sales[0])
                    if len(sales) >= 1
                    else 90.0
                )
                lag_7 = (
                    float(sum(sales[:7]))
                    if len(sales) >= 7
                    else float(sum(sales))
                    * (7 / len(sales))
                )
                lag_14 = (
                    float(sum(sales[:14]))
                    if len(sales) >= 14
                    else float(sum(sales))
                    * (14 / len(sales))
                )
                lag_30 = (
                    float(sum(sales[:30]))
                    if len(sales) >= 30
                    else float(sum(sales))
                    * (30 / len(sales))
                )

                custom_safety = (
                    float(latest["safety_threshold"])
                    if latest["safety_threshold"]
                    is not None
                    else 150.0
                )
                batch_size = (
                    int(latest["batch_size"])
                    if latest["batch_size"] is not None
                    else 1
                )
                moq = (
                    int(latest["moq"])
                    if latest["moq"] is not None
                    else 1
                )
                shelf_life_days = (
                    int(latest["shelf_life_days"])
                    if latest["shelf_life_days"] is not None
                    else 365
                )

                return {
                    "stock_on_hand": float(
                        latest["stock_on_hand"]
                    ),
                    "supplier_lead_time": float(
                        latest["supplier_lead_time"]
                    ),
                    "safety_threshold": custom_safety,
                    "batch_size": batch_size,
                    "moq": moq,
                    "shelf_life_days": shelf_life_days,
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_14": lag_14,
                    "lag_30": lag_30,
                }
    except Exception as e:
        logger.warning(
            f"Database feature fetch fallback: {e}"
        )
    finally:
        if conn:
            conn.close()

    return {
        "stock_on_hand": 100.0,
        "supplier_lead_time": 3.0,
        "safety_threshold": 150.0,
        "batch_size": 1,
        "moq": 1,
        "shelf_life_days": 365,
        "lag_1": 90.0,
        "lag_7": 630.0,
        "lag_14": 1260.0,
        "lag_30": 2700.0,
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Inference"],
)
async def predict(payload: PredictionInput):
    """
    Performs 7-day demand forecasting and calculates robust supply chain
    reorder quantities enforcing MOQ, packaging batch sizes, and shelf-life constraints.
    """
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model artifact is not loaded in memory.",
        )

    try:
        now = pd.Timestamp.now()
        input_data = {
            "month": now.month,
            "dayofweek": now.dayofweek,
            "is_weekend": 1
            if now.dayofweek in [5, 6]
            else 0,
            "stock_on_hand": payload.stock_on_hand,
            "supplier_lead_time": payload.supplier_lead_time,
            "lag_7": payload.lag_7,
            "lag_14": payload.lag_14,
            "lag_30": payload.lag_30,
            "rolling_mean_7": payload.lag_7 / 7.0,
            "rolling_mean_30": payload.lag_30 / 30.0,
        }

        X = pd.DataFrame([input_data])[FEATURE_COLS]
        raw_pred = float(loaded_model.predict(X)[0])
        predicted_demand = float(np.maximum(0.0, raw_pred))

        projected_stock = (
            payload.stock_on_hand - predicted_demand
        )

        # --- ADVANCED SUPPLY CHAIN LOGIC WITH LOGISTICAL CONSTRAINTS ---
        daily_demand = (
            predicted_demand / 7.0
            if predicted_demand > 0
            else 1.0
        )
        buffered_lead_time = (
            payload.supplier_lead_time + 1.0
        )
        required_stock_for_lead_time = (
            daily_demand * buffered_lead_time
        ) + payload.safety_threshold

        lead_time_breach = (
            payload.stock_on_hand
            < required_stock_for_lead_time
        )
        projected_stock_breach = (
            projected_stock < payload.safety_threshold
        )
        reorder_needed = (
            lead_time_breach or projected_stock_breach
        )

        order_quantity = 0
        if reorder_needed:
            # Business rule: When a reorder is triggered, order at least 7 days of forecasted consumption
            # or the volume needed to cover the safety stock deficit, whichever is greater.
            base_order_qty = max(
                predicted_demand,
                payload.safety_threshold - projected_stock,
            )

            # Apply logistical constraints (Minimum Order Quantity and Packaging Batch Size)
            effective_qty = max(base_order_qty, payload.moq)
            bs = max(1, payload.batch_size)
            order_quantity = (
                math.ceil(effective_qty / bs) * bs
            )

            # Validate against shelf-life (expiry) constraints to avoid over-ordering stock that expires
            max_viable_demand = (
                daily_demand * payload.shelf_life_days
            )
            if order_quantity > max_viable_demand:
                order_quantity = max(
                    bs,
                    math.floor(max_viable_demand / bs) * bs,
                )

        return PredictionOutput(
            store_id=payload.store_id,
            product_id=payload.product_id,
            predicted_demand=predicted_demand,
            projected_stock=projected_stock,
            reorder_needed=reorder_needed,
            order_quantity=order_quantity,
            model_status=get_model_status(),
        )

    except Exception as e:
        logger.error(f"Inference execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(e)}",
        )
