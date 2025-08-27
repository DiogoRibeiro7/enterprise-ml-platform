"""Online learning algorithms for streaming data."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier


class OnlineLearner:
    """Wrap a scikit-learn model for incremental updates."""

    def __init__(self, model: Optional[SGDClassifier] = None) -> None:
        self.model = model or SGDClassifier(loss="log_loss")
        self._classes: Optional[np.ndarray] = None

    async def partial_fit(
        self, features: Iterable[float], label: int, classes: Optional[Iterable[int]] = None
    ) -> None:
        X = np.asarray([list(features)])
        y = np.asarray([label])
        if self._classes is None:
            if classes is None:
                raise ValueError("classes must be provided on first call")
            self._classes = np.asarray(list(classes))
        self.model.partial_fit(X, y, classes=self._classes)

    async def predict(self, features: Iterable[float]) -> int:
        X = np.asarray([list(features)])
        return int(self.model.predict(X)[0])

    async def reset(self) -> None:
        self.model = SGDClassifier(loss="log_loss")
        self._classes = None
