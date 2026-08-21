"""Basic in-memory experiment tracking."""

from __future__ import annotations

from typing import Any


class ExperimentTracker:
    def __init__(self) -> None:
        self.experiments: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def log_experiment(self, params: dict[str, Any], metrics: dict[str, Any]) -> str:
        self._counter += 1
        exp_id = f"exp-{self._counter}"
        self.experiments[exp_id] = {"params": params, "metrics": metrics}
        return exp_id
