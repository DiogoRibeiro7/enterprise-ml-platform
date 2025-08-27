"""Prediction endpoints."""
from __future__ import annotations

from typing import List

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..dependencies import ModelRegistry, get_registry
from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    """Return a prediction for a single feature vector."""

    model = registry.get(request.model_name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    arr = np.array([request.features])
    preds = model.predict(arr).astype(float).tolist()
    background_tasks.add_task(lambda: None)
    return PredictionResponse(predictions=preds)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchPredictionResponse:
    """Return predictions for a batch of feature vectors."""

    model = registry.get(request.model_name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    arr = np.array(request.items)
    preds = model.predict(arr).astype(float).tolist()
    return BatchPredictionResponse(predictions=preds)
