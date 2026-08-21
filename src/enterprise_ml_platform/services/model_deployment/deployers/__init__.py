"""Cloud-specific deployer implementations.

Only AWS SageMaker is implemented. Deployers for other providers were removed
rather than left as stubs: a deployer that logs and returns a plausible URL
without calling anything is worse than no deployer, because it reports success.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseDeployer(Protocol):
    """Protocol for deployer implementations."""

    async def deploy(self, model_path: str, config: dict[str, Any]) -> str:
        """Deploy ``model_path`` and return the endpoint identifier."""

    async def update_traffic(self, endpoint: str, variants: dict[str, float]) -> None:
        """Update traffic distribution for A/B tests."""

    async def rollback(
        self, endpoint: str, previous_version: str | None = None
    ) -> None:
        """Rollback deployment to ``previous_version`` if provided."""

    async def delete(self, endpoint: str) -> None:
        """Remove deployment at ``endpoint``."""


from .aws_deployer import AWSDeployer, DeploymentError, EndpointState  # noqa: E402

__all__ = [
    "BaseDeployer",
    "AWSDeployer",
    "DeploymentError",
    "EndpointState",
]
