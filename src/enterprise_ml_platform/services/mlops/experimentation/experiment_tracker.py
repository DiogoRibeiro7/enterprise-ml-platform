"""Basic in-memory experiment tracking."""
from __future__ import annotations

from typing import Any, Dict


class ExperimentTracker:
    def __init__(self) -> None:
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def log_experiment(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        self._counter += 1
        exp_id = f"exp-{self._counter}"
        self.experiments[exp_id] = {"params": params, "metrics": metrics}
        return exp_id
