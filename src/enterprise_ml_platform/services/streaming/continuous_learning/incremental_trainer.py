"""Coordinate online learner updates in the streaming pipeline."""
from __future__ import annotations

from typing import Iterable, Optional

from .online_learner import OnlineLearner


class IncrementalTrainer:
    """Feed streaming examples to an :class:`OnlineLearner`."""

    def __init__(self, learner: OnlineLearner, classes: Optional[Iterable[int]] = None) -> None:
        self.learner = learner
        self.classes = list(classes) if classes is not None else None

    async def update(self, features: Iterable[float], label: int) -> None:
        await self.learner.partial_fit(features, label, classes=self.classes)
        self.classes = None  # classes only required on first update
