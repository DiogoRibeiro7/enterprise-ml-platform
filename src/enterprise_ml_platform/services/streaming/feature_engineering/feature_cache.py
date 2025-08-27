"""Low-latency in-memory feature cache."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class FeatureCache:
    """Simple TTL-based cache for derived features."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self.ttl = ttl_seconds
        self._store: Dict[str, tuple[float, Dict[str, Any]]] = {}

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        self._store[key] = (time.time(), value)
