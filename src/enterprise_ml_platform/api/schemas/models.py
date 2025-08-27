"""Schemas describing model metadata."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about a served model."""

    name: str
    version: str
    description: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
