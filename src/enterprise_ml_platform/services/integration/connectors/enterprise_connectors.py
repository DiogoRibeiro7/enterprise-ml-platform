"""Light-weight enterprise system connectors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnterpriseConnectors:
    """Registry of callables representing external systems."""

    connectors: dict[str, Callable[[dict[str, Any]], Any]] = field(default_factory=dict)

    def register(self, name: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.connectors[name] = handler

    def call(self, name: str, payload: dict[str, Any]) -> Any:
        if name not in self.connectors:
            raise KeyError(f"Unknown connector {name}")
        return self.connectors[name](payload)
