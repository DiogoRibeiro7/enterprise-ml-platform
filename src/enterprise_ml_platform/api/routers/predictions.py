"""Prediction endpoints.

Inference is CPU-bound and blocking. Running it directly inside an ``async``
handler stalls the event loop for the whole process, so every request -- not
just the one predicting -- waits behind it. All model calls here are therefore
dispatched to a worker thread.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..config import APISettings
from ..dependencies import (
    LoadedModel,
    ModelRegistry,
    get_registry,
    get_settings,
)
from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])


def _resolve(registry: ModelRegistry, name: str, version: str | None) -> LoadedModel:
    """Return the loaded model to serve, or fail with a useful status."""
    loaded = registry.get(name)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' is not loaded")
    if version is not None and version != loaded.version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{name}' is serving version {loaded.version}, "
                f"not the requested {version}"
            ),
        )
    return loaded


def _validate_shape(loaded: LoadedModel, rows: Sequence[Sequence[float]]) -> None:
    """Reject inputs the model cannot score before handing them over.

    scikit-learn raises on a feature-count mismatch, but a model that happens
    to tolerate it would score a misaligned vector and return a confident,
    meaningless answer.
    """
    if not rows:
        raise HTTPException(status_code=422, detail="No feature vectors supplied")
    widths = {len(row) for row in rows}
    if len(widths) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"All feature vectors must be the same length, got {sorted(widths)}",
        )
    width = widths.pop()
    expected = loaded.n_features
    if expected is not None and width != expected:
        raise HTTPException(
            status_code=422,
            detail=(f"Model '{loaded.name}' expects {expected} features, got {width}"),
        )


async def _predict(loaded: LoadedModel, rows: Sequence[Sequence[float]]) -> list[float]:
    """Score ``rows`` on a worker thread, keeping the event loop free."""
    array = np.asarray(rows, dtype=float)
    try:
        raw = await asyncio.to_thread(loaded.predict, array)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 400
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc
    return np.asarray(raw).astype(float).ravel().tolist()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: APISettings = Depends(get_settings),
) -> PredictionResponse:
    """Return a prediction for a single feature vector."""
    loaded = _resolve(registry, request.model_name, request.model_version)
    _validate_shape(loaded, [request.features])

    started = time.perf_counter()
    predictions = await _predict(loaded, [request.features])
    latency_ms = (time.perf_counter() - started) * 1000

    return PredictionResponse(
        predictions=predictions,
        model_name=loaded.name,
        model_version=loaded.version,
        latency_ms=latency_ms,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: APISettings = Depends(get_settings),
) -> BatchPredictionResponse:
    """Return predictions for a batch of feature vectors."""
    loaded = _resolve(registry, request.model_name, request.model_version)
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch of {len(request.items)} exceeds the limit of "
                f"{settings.max_batch_size}"
            ),
        )
    _validate_shape(loaded, request.items)

    started = time.perf_counter()
    predictions = await _predict(loaded, request.items)
    latency_ms = (time.perf_counter() - started) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        model_name=loaded.name,
        model_version=loaded.version,
        latency_ms=latency_ms,
    )
