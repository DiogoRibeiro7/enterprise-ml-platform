"""Light‑weight vision model abstractions used in the computer vision example."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class VisionModel:
    """Wrapper providing a unified interface for different vision tasks.

    Only a very small subset of functionality is implemented to keep the
    example self contained.  The model defaults to a logistic regression
    classifier and can be fine‑tuned to mimic transfer learning.
    """

    task: str = "classification"
    model: Any | None = None

    def load_pretrained(self, name: str | None = None) -> None:
        """Load a stub "pre‑trained" model."""
        # For demonstration we simply instantiate a new model; in a real system
        # this would download weights for models such as ResNet or YOLO.
        self.model = LogisticRegression(max_iter=100)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine‑tune the model on ``X``/``y``."""
        if self.model is None:
            self.load_pretrained()
        X = X.reshape(len(X), -1)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("model not trained")
        X = X.reshape(len(X), -1)
        return self.model.predict(X)
