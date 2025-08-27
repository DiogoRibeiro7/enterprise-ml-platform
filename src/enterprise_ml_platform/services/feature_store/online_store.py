from __future__ import annotations
"""Redis-backed online feature store."""

from typing import Dict, Iterable, Optional
import time

from redis.asyncio import Redis

from ..monitoring.collectors.metrics_collector import MetricsCollector


class OnlineFeatureStore:
    """Simple Redis feature cache for low-latency access."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl_seconds: int = 3600,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        self.metrics = metrics

    # ------------------------------------------------------------------
    async def set_features(self, entity_id: str, features: Dict[str, float]) -> None:
        await self.redis.hset(entity_id, mapping=features)
        await self.redis.expire(entity_id, self.ttl)

    # ------------------------------------------------------------------
    async def get_features(self, entity_id: str, names: Iterable[str]) -> Dict[str, float]:
        start = time.perf_counter()
        data = await self.redis.hmget(entity_id, *names)
        latency = time.perf_counter() - start
        hit = all(v is not None for v in data)
        if self.metrics:
            self.metrics.record_feature_serving(
                "online", latency, hit
            )
        if not hit:
            return {}
        return {n: float(v) for n, v in zip(names, data) if v is not None}

    # ------------------------------------------------------------------
    async def close(self) -> None:
        await self.redis.close()
