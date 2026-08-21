"""Canary deployment strategy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..deployers import BaseDeployer
from ..monitoring.health_checker import DeploymentHealthChecker


@dataclass
class CanaryStrategy:
    """Gradually shift traffic to a new deployment while monitoring health."""

    baseline_endpoint: str | None = None
    steps: int = 5
    delay: float = 1.0

    async def execute(
        self,
        deployer: BaseDeployer,
        model_path: str,
        health_checker: DeploymentHealthChecker,
    ) -> str:
        new_endpoint = await deployer.deploy(model_path, {})
        healthy = await health_checker.check(new_endpoint, deployer)
        if not healthy:
            await deployer.delete(new_endpoint)
            raise RuntimeError("Canary deployment failed health checks")

        if self.baseline_endpoint:
            for i in range(1, self.steps + 1):
                pct = i / self.steps
                await deployer.update_traffic(
                    endpoint=self.baseline_endpoint,
                    variants={self.baseline_endpoint: 1 - pct, new_endpoint: pct},
                )
                await asyncio.sleep(self.delay)
        self.baseline_endpoint = new_endpoint
        return new_endpoint
