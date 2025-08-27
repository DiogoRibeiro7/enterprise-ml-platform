"""Simple concept drift adapter for online learners."""
from __future__ import annotations

import collections
from typing import Deque

from .online_learner import OnlineLearner


class DriftAdapter:
    """Monitor prediction errors and reset learner on drift."""

    def __init__(self, learner: OnlineLearner, threshold: float = 0.5, window: int = 50) -> None:
        self.learner = learner
        self.threshold = threshold
        self.errors: Deque[int] = collections.deque(maxlen=window)

    async def report(self, truth: int, prediction: int) -> None:
        self.errors.append(int(truth != prediction))
        if len(self.errors) == self.errors.maxlen and sum(self.errors) / len(self.errors) > self.threshold:
            await self.learner.reset()
            self.errors.clear()
