"""Pydantic models for prediction endpoints."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for single predictions."""

    model_name: str = Field(..., description="Target model identifier")
    features: List[float] = Field(..., description="Feature vector for prediction")


class PredictionResponse(BaseModel):
    """Response model containing prediction results."""

    predictions: List[float]


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""

    model_name: str
    items: List[List[float]]


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction results."""

    predictions: List[float]
