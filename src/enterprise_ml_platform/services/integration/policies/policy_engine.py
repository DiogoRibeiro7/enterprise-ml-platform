"""Policy enforcement for the API gateway."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Policy = Callable[[dict[str, Any]], bool]


@dataclass
class PolicyEngine:
    """Stores and evaluates policy callables."""

    policies: dict[str, Policy] = field(default_factory=dict)

    def register(self, name: str, policy: Policy) -> None:
        self.policies[name] = policy

    def evaluate(self, request: dict[str, Any]) -> bool:
        return all(policy(request) for policy in self.policies.values())
