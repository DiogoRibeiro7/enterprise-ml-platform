"""Utilities for high‑throughput batch inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BatchInference:
    """Runs inference on images in configurable batches."""

    model: Any
    batch_size: int = 32

    def run(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X`` in batches."""
        outputs = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            outputs.append(self.model.predict(batch))
        return np.concatenate(outputs)
