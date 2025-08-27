"""API endpoints for managing A/B experiments."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from enterprise_ml_platform.services.ab_testing import (
    ExperimentConfig,
    ExperimentManager,
)

router = APIRouter(tags=["ab-testing"])
_manager = ExperimentManager()


@router.post("/ab-tests", status_code=201)
async def create_experiment(cfg: ExperimentConfig) -> dict:
    await _manager.create_experiment(cfg)
    return {"status": "created"}


@router.get("/ab-tests/{name}/assign")
async def assign(name: str, session_id: str) -> dict:
    try:
        variant = await _manager.get_variant(name, session_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc))
    return {"variant": variant}


@router.post("/ab-tests/{name}/outcome")
async def record_outcome(name: str, variant: str, value: float, success: bool) -> dict:
    await _manager.record_outcome(name, variant, value, success)
    return {"status": "recorded"}


@router.get("/ab-tests/{name}/analysis")
async def analysis(name: str) -> dict:
    return await _manager.analyze(name)
