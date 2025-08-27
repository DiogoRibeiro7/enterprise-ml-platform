from __future__ import annotations

"""Optimize workload scheduling order."""

from typing import List


class SchedulerOptimizer:
    def order(self, jobs: List[str]) -> List[str]:
        """Return jobs sorted alphabetically as a placeholder strategy."""
        return sorted(jobs)
