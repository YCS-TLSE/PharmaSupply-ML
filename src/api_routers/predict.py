"""
Demand inference and replenishment endpoint.

RESPONSIBILITIES
----------------
* Validate the inference payload through Pydantic.
* Build the feature vector with the shared train/serve builder when a sales
  history is provided, or with the legacy lag path otherwise.
* Run the XGBoost model, then apply inventory business rules (MOQ, batch
  rounding, shelf-life cap, FEFO loss).
* Record an aggregated prediction for production drift monitoring without
  blocking the response.

INPUTS / OUTPUTS
----------------
Inputs: ``PredictionInput`` (JSON body).
Outputs: ``PredictionOutput`` with predicted 7-day demand, projected stock,
reorder flag, order quantity, FEFO loss and model status.

ERROR CONTRACT
--------------
* ``503`` when the model artifact is not loaded.
* ``422`` for schema violations (Pydantic).
* ``400`` for unexpected inference errors.
"""

import logging

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    status,
)

import src.config as config
from src.config import FEATURE_COLS
from src.observability import record_prediction
from src.schemas import PredictionInput, PredictionOutput
from src.security import limiter, require_api_key
from src.services.inventory import (
    build_prediction_features,
    compute_fefo_loss,
    compute_reorder,
)
from src.services.model_status import read_model_status

logger = logging.getLogger("pharmasupply.predict")

router = APIRouter(
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)


def get_loaded_model(request: Request):
    """Dependency exposing the model loaded during application startup.

    Exposing the model as a dependency (instead of importing the global)
    allows tests to override it with a deterministic stub.
    """
    return getattr(request.app.state, "model", None)


@router.post("/predict", response_model=PredictionOutput)
@limiter.limit(config.RATE_LIMIT_PREDICT)
def predict(
    request: Request,
    payload: PredictionInput,
    background_tasks: BackgroundTasks,
    model=Depends(get_loaded_model),
) -> PredictionOutput:
    """Predict 7-day demand and compute a constraint-compliant order quantity.

    When ``payload.history`` is provided, the shared feature builder is used,
    guaranteeing the exact same transformations as the training pipeline.
    """
    # The model is loaded once in the lifespan; absence means the artifact was
    # missing or failed to deserialise, so inference cannot be served.
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model artifact is not loaded in memory.",
        )

    try:
        # 1. Feature construction: parity-safe when history is available.
        features = build_prediction_features(payload)
        X = pd.DataFrame([features])[list(FEATURE_COLS)]

        # 2. Model inference. Negative predictions are clipped to zero because
        #    demand cannot be negative.
        raw_prediction = float(model.predict(X)[0])
        predicted_demand = max(0.0, raw_prediction)

        # 3. Replenishment rules: lead-time coverage, safety stock, MOQ,
        #    batch multiples and shelf-life cap are all encapsulated in the
        #    pure service function.
        decision = compute_reorder(
            predicted_demand=predicted_demand,
            stock_on_hand=payload.stock_on_hand,
            safety_threshold=payload.safety_threshold,
            supplier_lead_time=payload.supplier_lead_time,
            moq=payload.moq,
            batch_size=payload.batch_size,
            shelf_life_days=payload.shelf_life_days,
        )

        # 4. FEFO loss: demand lost to lots expiring inside the horizon.
        fefo_expired_loss = compute_fefo_loss(
            predicted_demand, payload.batches
        )

        # 5. Production drift input: aggregated features + prediction, written
        #    asynchronously so disk I/O does not affect response latency.
        background_tasks.add_task(
            record_prediction,
            features,
            predicted_demand,
            decision.reorder_needed,
            payload.store_id,
            payload.product_id,
        )

        return PredictionOutput(
            store_id=payload.store_id,
            product_id=payload.product_id,
            predicted_demand=predicted_demand,
            projected_stock=decision.projected_stock,
            reorder_needed=decision.reorder_needed,
            order_quantity=decision.order_quantity,
            fefo_expired_loss=fefo_expired_loss,
            model_status=read_model_status(),
        )
    except HTTPException:
        # Preserve explicit HTTP errors raised above.
        raise
    except Exception as exc:
        # Any other failure is a client-visible inference error. The exception
        # is logged with its traceback for diagnosis.
        logger.exception(
            "Inference execution error: %s", exc
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(exc)}",
        ) from exc
