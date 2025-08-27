"""Light-weight enterprise system connectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class EnterpriseConnectors:
    """Registry of callables representing external systems."""

    connectors: Dict[str, Callable[[Dict[str, Any]], Any]] = field(default_factory=dict)

    def register(self, name: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self.connectors[name] = handler

    def call(self, name: str, payload: Dict[str, Any]) -> Any:
        if name not in self.connectors:
            raise KeyError(f"Unknown connector {name}")
        return self.connectors[name](payload)
