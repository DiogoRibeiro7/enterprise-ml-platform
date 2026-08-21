"""Simple deployment automation utilities."""

from __future__ import annotations

from typing import Any


class DeploymentAutomator:
    """Store deployments in memory keyed by environment."""

    def __init__(self) -> None:
        self.deployments: dict[str, Any] = {}

    def deploy(self, model: Any, environment: str = "dev") -> str:
        identifier = f"{environment}-{len(self.deployments) + 1}"
        self.deployments[identifier] = model
        return identifier
