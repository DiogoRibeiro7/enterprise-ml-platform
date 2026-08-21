"""Simple Thompson sampling for Bernoulli rewards."""

from __future__ import annotations

import numpy as np


class ThompsonSamplingStrategy:
    def __init__(self) -> None:
        self.successes: dict[str, int] = {}
        self.failures: dict[str, int] = {}

    def select(self, variants: dict[str, float]) -> str:
        for v in variants:
            self.successes.setdefault(v, 1)
            self.failures.setdefault(v, 1)
        samples = {
            v: np.random.beta(self.successes[v], self.failures[v]) for v in variants
        }
        return max(samples, key=samples.get)

    def update(self, variant: str, success: bool) -> None:
        if success:
            self.successes[variant] += 1
        else:
            self.failures[variant] += 1
