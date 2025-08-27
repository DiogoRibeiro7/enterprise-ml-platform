from __future__ import annotations

"""Orchestrator for model deployments across multiple cloud providers."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import structlog

from .deployers import (
    AWSDeployer,
    AzureDeployer,
    GCPDeployer,
    BaseDeployer,
)
from .strategies import (
    BlueGreenStrategy,
    CanaryStrategy,
    RollingStrategy,
    DeploymentStrategy,
)
from .monitoring.health_checker import DeploymentHealthChecker
from .rollback.rollback_manager import RollbackManager

logger = structlog.get_logger()


@dataclass
class DeploymentConfig:
    """Configuration options for a model deployment."""

    platform: str
    strategy: str = "blue_green"
    platform_config: Dict[str, Any] = field(default_factory=dict)
    strategy_config: Dict[str, Any] = field(default_factory=dict)


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
        traffic_split: Optional[Dict[str, float]] = None,
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
    def _build_deployer(self, platform: str, cfg: Dict[str, Any]) -> BaseDeployer:
        platform = platform.lower()
        if platform == "aws":
            return AWSDeployer(**cfg)
        if platform == "gcp":
            return GCPDeployer(**cfg)
        if platform == "azure":
            return AzureDeployer(**cfg)
        raise ValueError(f"Unsupported platform: {platform}")

    def _build_strategy(self, name: str, cfg: Dict[str, Any]) -> DeploymentStrategy:
        name = name.replace("-", "_").lower()
        if name == "blue_green" or name == "bluegreen":
            return BlueGreenStrategy(**cfg)
        if name == "canary":
            return CanaryStrategy(**cfg)
        if name == "rolling":
            return RollingStrategy(**cfg)
        raise ValueError(f"Unknown strategy: {name}")
