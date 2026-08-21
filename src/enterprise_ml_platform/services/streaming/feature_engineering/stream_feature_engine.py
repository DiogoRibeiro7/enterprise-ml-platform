"""Orchestrate real-time feature computation and enrichment."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .feature_cache import FeatureCache
from .stream_joins import StreamJoiner
from .window_operations import CountWindowAggregator, TimeWindowAggregator


class StreamFeatureEngine:
    """Compute derived streaming features with low latency."""

    def __init__(
        self,
        window_ops: Iterable[TimeWindowAggregator | CountWindowAggregator]
        | None = None,
        joiners: Iterable[StreamJoiner] | None = None,
        cache: FeatureCache | None = None,
    ) -> None:
        self.window_ops = list(window_ops or [])
        self.joiners = list(joiners or [])
        self.cache = cache or FeatureCache()

    async def compute(self, features: dict[str, Any]) -> dict[str, Any]:
        key = features.get("entity_id")
        cached = await self.cache.get(key) if key else None
        base = cached.copy() if cached else {}
        base.update(features)
        for joiner in self.joiners:
            base = await joiner.join(base)
        for op in self.window_ops:
            base.update(await op.apply(base))
        if key:
            await self.cache.set(key, base)
        return base
