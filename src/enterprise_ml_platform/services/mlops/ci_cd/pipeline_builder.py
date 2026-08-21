"""Utilities for constructing CI/CD pipelines.

The real platform would parse workflow definitions from Git repositories and
translate them into DAGs.  For the purposes of the kata the builder simply
wraps a list of callables representing the sequential pipeline steps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

Step = Callable[[object], object]


class PipelineBuilder:
    """Create ordered pipelines from an iterable of steps."""

    def build(self, steps: Iterable[Step]) -> list[Step]:
        return list(steps)
