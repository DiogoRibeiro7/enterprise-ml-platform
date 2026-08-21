"""Recommend instance sizes based on historical usage."""

from __future__ import annotations

from collections.abc import Iterable


class Rightsizer:
    def recommend(self, usage: Iterable[float]) -> float:
        """Return recommended instance size (simple average)."""
        data = list(usage)
        if not data:
            return 0.0
        return sum(data) / len(data)
