"""Schemas describing model metadata."""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about a served model."""

    name: str
    version: str
    description: str | None = None
    metrics: dict[str, float] | None = None
