"""Multi-armed bandit strategy using epsilon-greedy."""

from __future__ import annotations

import random


class EpsilonGreedyBandit:
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.counts: dict[str, int] = {}
        self.values: dict[str, float] = {}

    def select(self, variants: dict[str, float]) -> str:
        for v in variants:
            self.counts.setdefault(v, 0)
            self.values.setdefault(v, 0.0)
        if random.random() < self.epsilon:
            return random.choice(list(variants.keys()))
        return max(self.values, key=self.values.get)

    def update(self, variant: str, reward: float) -> None:
        self.counts[variant] += 1
        n = self.counts[variant]
        value = self.values[variant]
        self.values[variant] = value + (reward - value) / n
