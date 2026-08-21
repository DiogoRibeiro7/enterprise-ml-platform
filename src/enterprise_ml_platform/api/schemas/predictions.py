"""Pydantic models for prediction endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for single predictions."""

    model_name: str = Field(..., description="Target model identifier")
    features: list[float] = Field(..., description="Feature vector for prediction")
    model_version: str | None = Field(
        default=None,
        description="Pin the request to one version instead of the served alias",
    )


class PredictionResponse(BaseModel):
    """Response model containing prediction results.

    The version is echoed back so a caller can tell which model produced a
    prediction. Without it, a promotion silently changes the answers.
    """

    predictions: list[float]
    model_name: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""

    model_name: str
    items: list[list[float]]
    model_version: str | None = Field(
        default=None,
        description="Pin the request to one version instead of the served alias",
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction results."""

    predictions: list[float]
    model_name: str
    model_version: str
    latency_ms: float
