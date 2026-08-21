"""Rolling deployment strategy."""

from __future__ import annotations

from dataclasses import dataclass

from ..deployers import BaseDeployer
from ..monitoring.health_checker import DeploymentHealthChecker


@dataclass
class RollingStrategy:
    """Replace instances in batches to minimise downtime."""

    batch_size: int = 1

    async def execute(
        self,
        deployer: BaseDeployer,
        model_path: str,
        health_checker: DeploymentHealthChecker,
    ) -> str:
        endpoint = await deployer.deploy(model_path, {})
        healthy = await health_checker.check(endpoint, deployer)
        if not healthy:
            await deployer.delete(endpoint)
            raise RuntimeError("Rolling deployment failed health checks")
        return endpoint
