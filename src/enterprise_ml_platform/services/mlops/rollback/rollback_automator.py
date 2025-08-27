"""Utilities for reverting deployments."""
from __future__ import annotations

from typing import List, Optional


class RollbackAutomator:
    def __init__(self) -> None:
        self.history: List[str] = []

    def checkpoint(self, deployment_id: str) -> None:
        self.history.append(deployment_id)

    def rollback(self) -> Optional[str]:
        if self.history:
            return self.history.pop()
        return None
