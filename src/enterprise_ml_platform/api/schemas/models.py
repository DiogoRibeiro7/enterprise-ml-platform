"""Schemas describing model metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about a served model."""

    name: str
    version: str
    description: str | None = None
    metrics: dict[str, float] | None = None


class DriftStatus(BaseModel):
    """Live input-drift state for one served model version."""

    model_name: str
    model_version: str
    state: Literal["unavailable", "collecting", "ready"]
    observed_rows: int
    required_rows: int
    window_size: int
    threshold: float
    scores: dict[str, float]
    drifted_features: list[str]
