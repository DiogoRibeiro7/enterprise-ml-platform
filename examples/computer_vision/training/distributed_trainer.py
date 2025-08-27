"""Toy distributed training helper for vision models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DistributedTrainer:
    """Simulates multi‑worker training by sharding the data.

    The goal is to mirror how frameworks like PyTorch Distributed or Horovod
    would orchestrate training across GPUs without depending on those heavy
    libraries for this example.
    """

    model: Any
    workers: int = 1

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Shard ``X`` across ``workers`` and train the wrapped model."""
        if self.workers <= 1:
            self.model.train(X, y)
            return

        splits = np.array_split(X, self.workers)
        ys = np.array_split(y, self.workers)
        for split, target in zip(splits, ys):
            # In real distributed training each shard would be processed on a
            # different device and gradients aggregated.  Here we simply call
            # train sequentially to emulate the behaviour.
            self.model.train(split, target)
