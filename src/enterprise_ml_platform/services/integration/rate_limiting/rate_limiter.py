"""Token bucket based rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Enforces per-user request limits."""

    rate: int
    per: float
    buckets: dict[str, tuple[float, float]] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        tokens, last = self.buckets.get(key, (float(self.rate), time.time()))
        now = time.time()
        tokens = min(self.rate, tokens + (now - last) * self.rate / self.per)
        if tokens < 1:
            self.buckets[key] = (tokens, now)
            return False
        self.buckets[key] = (tokens - 1, now)
        return True
