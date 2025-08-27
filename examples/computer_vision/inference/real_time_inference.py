"""Low‑latency real‑time inference pipeline."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RealTimeInference:
    """Queue based real‑time inference helper."""

    model: Any
    queue: deque[np.ndarray] = field(default_factory=deque)

    def submit(self, image: np.ndarray) -> None:
        """Submit an ``image`` for inference."""
        self.queue.append(image)

    def process_next(self) -> Any:
        """Process the next image in the queue."""
        if not self.queue:
            return None
        img = self.queue.popleft()
        return self.model.predict(np.expand_dims(img, 0))[0]
