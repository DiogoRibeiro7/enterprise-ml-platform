"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...services.monitoring.serving_drift import DriftReport, ServingDriftMonitor
from ..dependencies import (
    ModelNotAvailableError,
    ModelRegistry,
    get_drift_monitor,
    get_registry,
)
from ..schemas import DriftStatus, ModelInfo

router = APIRouter(tags=["models"])


def _drift_status(report: DriftReport) -> DriftStatus:
    return DriftStatus(
        model_name=report.model_name,
        model_version=report.model_version,
        state=report.state,
        observed_rows=report.observed_rows,
        required_rows=report.required_rows,
        window_size=report.window_size,
        threshold=report.threshold,
        scores=report.scores,
        drifted_features=list(report.drifted_features),
    )


@router.get("/models", response_model=list[str])
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> list[str]:
    """List names of loaded models."""

    return registry.list_models()


@router.get("/models/{model_name}/info", response_model=ModelInfo)
async def model_info(
    model_name: str, registry: ModelRegistry = Depends(get_registry)
) -> ModelInfo:
    """Return information about a model."""

    try:
        return registry.info(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/{model_name}/load", response_model=ModelInfo)
async def load_model(
    model_name: str,
    alias: str | None = None,
    version: str | None = None,
    registry: ModelRegistry = Depends(get_registry),
    drift_monitor: ServingDriftMonitor = Depends(get_drift_monitor),
) -> ModelInfo:
    """Load a model into the serving cache.

    Resolves the configured alias by default, so the deployed champion is
    served without naming a version.
    """

    previous = registry.get(model_name)
    try:
        info = registry.load(model_name, alias=alias, version=version)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    loaded = registry.get(model_name)
    assert loaded is not None
    if loaded.drift_reference is not None:
        drift_monitor.register(loaded.name, loaded.version, loaded.drift_reference)
    else:
        drift_monitor.remove(loaded.name, loaded.version)
    if previous is not None and previous.version != loaded.version:
        drift_monitor.remove(previous.name, previous.version)
    return info


@router.get("/models/{model_name}/drift", response_model=DriftStatus)
async def model_drift(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
    drift_monitor: ServingDriftMonitor = Depends(get_drift_monitor),
) -> DriftStatus:
    """Return live input-drift state for the exact served artifact."""
    loaded = registry.get(model_name)
    if loaded is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' is not loaded"
        )
    return _drift_status(
        drift_monitor.status(
            loaded.name, loaded.version, reference=loaded.drift_reference
        )
    )


@router.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
    drift_monitor: ServingDriftMonitor = Depends(get_drift_monitor),
) -> dict:
    """Unload a model from memory."""

    loaded = registry.get(model_name)
    registry.unload(model_name)
    if loaded is not None:
        drift_monitor.remove(loaded.name, loaded.version)
    return {"status": "unloaded", "model_name": model_name}
