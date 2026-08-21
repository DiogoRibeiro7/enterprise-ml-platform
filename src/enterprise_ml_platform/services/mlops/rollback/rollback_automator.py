"""Utilities for reverting deployments."""

from __future__ import annotations


class RollbackAutomator:
    def __init__(self) -> None:
        self.history: list[str] = []

    def checkpoint(self, deployment_id: str) -> None:
        self.history.append(deployment_id)

    def rollback(self) -> str | None:
        if self.history:
            return self.history.pop()
        return None
