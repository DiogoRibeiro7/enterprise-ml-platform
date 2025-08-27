from __future__ import annotations

"""Blue-green deployment strategy."""

from dataclasses import dataclass
from typing import Optional

from ..deployers import BaseDeployer
from ..monitoring.health_checker import DeploymentHealthChecker


@dataclass
class BlueGreenStrategy:
    """Deploy new version alongside current and switch traffic when healthy."""

    current_endpoint: Optional[str] = None

    async def execute(
        self,
        deployer: BaseDeployer,
        model_path: str,
        health_checker: DeploymentHealthChecker,
    ) -> str:
        new_endpoint = await deployer.deploy(model_path, {})
        healthy = await health_checker.check(new_endpoint)
        if not healthy:
            await deployer.delete(new_endpoint)
            raise RuntimeError("New deployment failed health checks")
        if self.current_endpoint:
            await deployer.update_traffic(new_endpoint, {"new": 1.0})
            await deployer.delete(self.current_endpoint)
        self.current_endpoint = new_endpoint
        return new_endpoint
