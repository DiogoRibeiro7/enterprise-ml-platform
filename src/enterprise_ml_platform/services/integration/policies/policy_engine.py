"""Policy enforcement for the API gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

Policy = Callable[[Dict[str, Any]], bool]


@dataclass
class PolicyEngine:
    """Stores and evaluates policy callables."""

    policies: Dict[str, Policy] = field(default_factory=dict)

    def register(self, name: str, policy: Policy) -> None:
        self.policies[name] = policy

    def evaluate(self, request: Dict[str, Any]) -> bool:
        return all(policy(request) for policy in self.policies.values())
