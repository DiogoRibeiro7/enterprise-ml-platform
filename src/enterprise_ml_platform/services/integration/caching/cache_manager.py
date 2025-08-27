"""In-memory cache manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import time


@dataclass
class CacheManager:
    """Very small TTL based cache used by the gateway."""

    store: Dict[str, tuple[Any, float]] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        item = self.store.get(key)
        if not item:
            return None
        value, expiry = item
        if expiry and expiry < time.time():
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        self.store[key] = (value, time.time() + ttl)
