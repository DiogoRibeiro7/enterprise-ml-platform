"""Orchestrator for model deployments across multiple cloud providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from .deployers import (
    AWSDeployer,
    BaseDeployer,
)
from .monitoring.health_checker import DeploymentHealthChecker
from .rollback.rollback_manager import RollbackManager
from .strategies import (
    BlueGreenStrategy,
    CanaryStrategy,
    DeploymentStrategy,
    RollingStrategy,
)

logger = structlog.get_logger()


@dataclass
class DeploymentConfig:
    """Configuration options for a model deployment."""

    platform: str
    strategy: str = "blue_green"
    platform_config: dict[str, Any] = field(default_factory=dict)
    strategy_config: dict[str, Any] = field(default_factory=dict)


class ModelDeploymentService:
    """Coordinate model deployments using pluggable deployers and strategies."""

    def __init__(self) -> None:
        self.logger = logger.bind(service="model-deployment")
        self.health_checker = DeploymentHealthChecker()
        self.rollback_manager = RollbackManager()

    async def deploy_model(
        self,
        model_path: str,
        config: DeploymentConfig,
        traffic_split: dict[str, float] | None = None,
    ) -> str:
        """Deploy a model according to ``config``.

        Args:
            model_path: Path or identifier of the model artifact.
            config: Deployment configuration.
            traffic_split: Optional mapping of variant identifiers to traffic
                percentages for A/B testing.

        Returns:
            URL of the deployed endpoint.
        """
        deployer = self._build_deployer(config.platform, config.platform_config)
        strategy = self._build_strategy(config.strategy, config.strategy_config)
        endpoint = await strategy.execute(
            deployer,
            model_path,
            self.health_checker,
        )
        if traffic_split:
            await deployer.update_traffic(endpoint, traffic_split)
        self.rollback_manager.register(endpoint, deployer)
        return endpoint

    async def rollback(self, endpoint: str) -> None:
        """Trigger rollback for ``endpoint`` using stored history."""
        await self.rollback_manager.rollback(endpoint)

    # ------------------------------------------------------------------
    def _build_deployer(self, platform: str, cfg: dict[str, Any]) -> BaseDeployer:
        platform = platform.lower()
        if platform == "aws":
            return AWSDeployer(**cfg)
        raise ValueError(
            f"Unsupported platform: {platform}. Only 'aws' is implemented."
        )

    def _build_strategy(self, name: str, cfg: dict[str, Any]) -> DeploymentStrategy:
        name = name.replace("-", "_").lower()
        if name == "blue_green" or name == "bluegreen":
            return BlueGreenStrategy(**cfg)
        if name == "canary":
            return CanaryStrategy(**cfg)
        if name == "rolling":
            return RollingStrategy(**cfg)
        raise ValueError(f"Unknown strategy: {name}")
