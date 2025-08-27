"""Warm up online learners with historical data."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .online_learner import OnlineLearner


class ModelWarmer:
    """Pre-train a learner before streaming begins."""

    def __init__(
        self,
        learner: OnlineLearner,
        warm_data: Sequence[Tuple[Iterable[float], int]],
        classes: Iterable[int],
    ) -> None:
        self.learner = learner
        self.warm_data = warm_data
        self.classes = list(classes)

    async def warm(self) -> None:
        classes = self.classes
        for features, label in self.warm_data:
            await self.learner.partial_fit(features, label, classes=classes)
            classes = None  # only supply on first call
