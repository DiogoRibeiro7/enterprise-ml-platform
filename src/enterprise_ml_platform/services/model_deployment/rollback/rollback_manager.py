"""Automated rollback utilities for deployments."""

from __future__ import annotations

import structlog

from ..deployers import BaseDeployer

logger = structlog.get_logger()


class RollbackManager:
    """Track deployments and trigger rollbacks when necessary."""

    def __init__(self) -> None:
        self._history: dict[str, tuple[BaseDeployer, str | None]] = {}
        self.logger = logger.bind(component="rollback-manager")

    def register(
        self, endpoint: str, deployer: BaseDeployer, previous_version: str | None = None
    ) -> None:
        """Register a deployment for potential rollback."""
        self._history[endpoint] = (deployer, previous_version)

    async def rollback(self, endpoint: str) -> None:
        """Rollback the deployment associated with ``endpoint`` if known."""
        entry = self._history.get(endpoint)
        if not entry:
            self.logger.warning("unknown-endpoint", endpoint=endpoint)
            return
        deployer, version = entry
        await deployer.rollback(endpoint, version)
