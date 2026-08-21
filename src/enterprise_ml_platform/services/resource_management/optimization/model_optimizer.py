"""Placeholder for model compression logic."""

from __future__ import annotations


class ModelOptimizer:
    def compress(self, size_mb: float) -> float:
        """Return a compressed size using a naive ratio."""
        return size_mb * 0.5
