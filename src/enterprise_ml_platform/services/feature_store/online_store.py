"""Redis-backed online feature store."""

from __future__ import annotations

import time
from collections.abc import Iterable

from redis.asyncio import Redis

from ..monitoring.collectors.metrics_collector import MetricsCollector

KEY_PREFIX = "features"


def build_key(feature_set: str, version: str, entity_id: str) -> str:
    """Return the Redis key holding ``entity_id`` for one feature set version.

    Entity ids are only unique *within* a feature set, and the same entity
    carries different values across versions. Keying on the entity id alone
    makes unrelated feature sets overwrite each other.
    """
    return f"{KEY_PREFIX}:{feature_set}:{version}:{entity_id}"


class OnlineFeatureStore:
    """Low-latency Redis cache for serving features."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl_seconds: int = 3600,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        self.metrics = metrics

    # ------------------------------------------------------------------
    async def set_features(
        self,
        feature_set: str,
        version: str,
        entity_id: str,
        features: dict[str, float],
    ) -> None:
        """Cache ``features`` for one entity of ``feature_set`` at ``version``.

        Values are coerced to ``float`` because that is what :meth:`get_features`
        reads back, and because Redis rejects Python ``bool`` outright -- a
        boolean feature would otherwise fail at write time.

        Raises:
            TypeError: If a feature value is not numeric.
        """
        if not features:
            return
        encoded: dict[str, float] = {}
        for name, value in features.items():
            try:
                encoded[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"feature '{name}' of feature set '{feature_set}' is not "
                    f"numeric: {value!r}"
                ) from exc
        key = build_key(feature_set, version, entity_id)
        await self.redis.hset(key, mapping=encoded)  # type: ignore[misc]
        await self.redis.expire(key, self.ttl)

    # ------------------------------------------------------------------
    async def get_features(
        self,
        feature_set: str,
        version: str,
        entity_id: str,
        names: Iterable[str],
    ) -> dict[str, float]:
        """Return the requested features, or ``{}`` on a partial or full miss."""
        names = list(names)
        if not names:
            return {}
        key = build_key(feature_set, version, entity_id)
        start = time.perf_counter()
        data = await self.redis.hmget(key, names)  # type: ignore[misc]
        latency = time.perf_counter() - start
        hit = bool(data) and all(v is not None for v in data)
        if self.metrics:
            self.metrics.record_feature_serving("online", latency, hit)
        if not hit:
            return {}
        return {n: float(v) for n, v in zip(names, data, strict=True) if v is not None}

    # ------------------------------------------------------------------
    async def close(self) -> None:
        await self.redis.aclose()
