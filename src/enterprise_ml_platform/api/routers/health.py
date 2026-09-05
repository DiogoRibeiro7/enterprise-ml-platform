"""Health check and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ...services.monitoring.collectors import MetricsCollector
from ..dependencies import ModelRegistry, get_metrics, get_registry
from ..schemas import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Basic health check."""

    return HealthStatus(status="ok")


@router.get("/health/detailed", response_model=HealthStatus)
async def health_detailed(
    registry: ModelRegistry = Depends(get_registry),
) -> HealthStatus:
    """Detailed health check with model information."""

    return HealthStatus(
        status="ok", details={"models_loaded": str(len(registry.list_models()))}
    )


@router.get("/metrics")
async def metrics(
    collector: MetricsCollector = Depends(get_metrics),
) -> Response:
    """Expose Prometheus metrics."""

    data = generate_latest(collector.registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
