"""Deployment strategy implementations."""

from __future__ import annotations

from typing import Protocol

from ..deployers import BaseDeployer
from ..monitoring.health_checker import DeploymentHealthChecker


class DeploymentStrategy(Protocol):
    """Protocol for deployment strategies."""

    async def execute(
        self,
        deployer: BaseDeployer,
        model_path: str,
        health_checker: DeploymentHealthChecker,
    ) -> str:
        """Execute the strategy and return the deployed endpoint URL."""


from .blue_green import BlueGreenStrategy
from .canary import CanaryStrategy
from .rolling import RollingStrategy

__all__ = [
    "DeploymentStrategy",
    "BlueGreenStrategy",
    "CanaryStrategy",
    "RollingStrategy",
]
