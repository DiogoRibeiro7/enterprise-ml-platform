"""Track data quality metrics over time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class QualityMonitor:
    """Store recent quality scores for trend analysis."""

    window: int = 100
    history: deque[float] = field(default_factory=deque)

    def record(self, score: float) -> None:
        """Record a new quality ``score``."""

        self.history.append(score)
        while len(self.history) > self.window:
            self.history.popleft()

    def trend(self) -> float:
        """Return simple moving average of recorded scores."""

        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)
