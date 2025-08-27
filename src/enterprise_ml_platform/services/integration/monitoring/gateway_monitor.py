"""Monitoring helpers for the API gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import DefaultDict


@dataclass
class GatewayMonitor:
    """Tracks request counts and errors for observability."""

    requests: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_request(self, route: str) -> None:
        self.requests[route] += 1

    def record_error(self, route: str) -> None:
        self.errors[route] += 1
