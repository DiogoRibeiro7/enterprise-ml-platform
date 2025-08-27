"""Simple request routing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RequestRouter:
    """Maps request paths to connector names with optional versioning."""

    routes: Dict[str, str] = field(default_factory=dict)

    def register(self, path_prefix: str, connector: str) -> None:
        self.routes[path_prefix] = connector

    def route(self, request: Dict[str, object]) -> str:
        path = str(request.get("path", "/"))
        version = request.get("version")
        for prefix, connector in self.routes.items():
            full_prefix = f"/{version}{prefix}" if version else prefix
            if path.startswith(full_prefix):
                return connector
        raise KeyError(f"No route for {path}")
