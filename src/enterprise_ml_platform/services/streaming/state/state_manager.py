"""State management utilities for streaming pipelines."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class StateManager:
    """Simple in-memory state store with async interface."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self.logger = logger.bind(component="state-manager")

    async def get(self, key: str) -> Any:
        return self._state.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def update_state(self, event: dict[str, Any]) -> None:
        """Example state update using event data."""
        if "id" in event:
            await self.set(event["id"], event)

    async def close(self) -> None:
        self._state.clear()
        self.logger.info("state-cleared")
