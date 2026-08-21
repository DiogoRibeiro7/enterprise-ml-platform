"""API endpoints for feature store operations."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...services.feature_store import FeatureStoreService
from ..dependencies import get_feature_store

router = APIRouter(prefix="/feature-store", tags=["feature-store"])


@router.get("/stats")
async def stats(
    service: FeatureStoreService = Depends(get_feature_store),
) -> dict:
    """Return basic feature store metrics."""

    metrics = service.metrics
    hits = metrics.feature_cache_hits.labels("online")._value.get()
    misses = metrics.feature_cache_misses.labels("online")._value.get()
    return {"cache_hits": hits, "cache_misses": misses}
