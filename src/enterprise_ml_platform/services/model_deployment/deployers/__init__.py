"""Cloud-specific deployer implementations."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class BaseDeployer(Protocol):
    """Protocol for deployer implementations."""

    async def deploy(self, model_path: str, config: Dict[str, Any]) -> str:
        """Deploy ``model_path`` and return the endpoint URL."""

    async def update_traffic(self, endpoint: str, variants: Dict[str, float]) -> None:
        """Update traffic distribution for A/B tests."""

    async def rollback(self, endpoint: str, previous_version: str | None = None) -> None:
        """Rollback deployment to ``previous_version`` if provided."""

    async def delete(self, endpoint: str) -> None:
        """Remove deployment at ``endpoint``."""


from .aws_deployer import AWSDeployer
from .gcp_deployer import GCPDeployer
from .azure_deployer import AzureDeployer

__all__ = [
    "BaseDeployer",
    "AWSDeployer",
    "GCPDeployer",
    "AzureDeployer",
]
