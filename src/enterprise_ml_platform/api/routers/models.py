"""Model management endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import ModelRegistry, get_registry
from ..schemas import ModelInfo

router = APIRouter(tags=["models"])


@router.get("/models", response_model=List[str])
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> List[str]:
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
    model_name: str, registry: ModelRegistry = Depends(get_registry)
) -> ModelInfo:
    """Load a model into memory."""

    return registry.load(model_name)


@router.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str, registry: ModelRegistry = Depends(get_registry)
) -> dict:
    """Unload a model from memory."""

    registry.unload(model_name)
    return {"status": "unloaded", "model_name": model_name}
