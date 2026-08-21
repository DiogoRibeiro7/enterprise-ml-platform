"""Deployment health monitoring utilities."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib import request
from urllib.parse import urlparse

import structlog

ALLOWED_HEALTH_CHECK_SCHEMES = frozenset({"http", "https"})

logger = structlog.get_logger()


class DeploymentHealthChecker:
    """Decide whether a deployment is serving.

    How that question is answered depends on what was deployed. A plain HTTP
    service can be fetched; a managed endpoint cannot -- it is invoked through
    a signed API and its readiness lives in the provider's control plane. When
    the deployer knows how to answer, it is asked; otherwise the target is
    treated as a URL.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        """Create a checker with an HTTP timeout for URL targets."""
        self.timeout = timeout
        self.logger = logger.bind(component="health-checker")

    async def check(self, target: str, deployer: Any | None = None) -> bool:
        """Return whether ``target`` is healthy.

        Args:
            target: Endpoint identifier, or a URL when no deployer is given.
            deployer: Deployer that created the endpoint. Used when it exposes
                ``check_health``.
        """
        if deployer is not None and hasattr(deployer, "check_health"):
            healthy: bool = await deployer.check_health(target)
            return healthy
        return await self.check_url(target)

    async def check_url(self, url: str) -> bool:
        """Return ``True`` if ``url`` responds with status < 400.

        Only http and https are accepted. ``urlopen`` also handles ``file:``
        and ``ftp:``, so an unvalidated endpoint string could turn a health
        check into a local file read.
        """
        scheme = urlparse(url).scheme.lower()
        if scheme not in ALLOWED_HEALTH_CHECK_SCHEMES:
            self.logger.warning("health-check-scheme-rejected", url=url, scheme=scheme)
            return False

        def _check() -> bool:
            try:
                # Scheme is checked against an allow-list above. # nosec B310
                with request.urlopen(url, timeout=self.timeout) as resp:  # nosec B310
                    status: int = resp.status
                return status < 400
            except Exception as exc:  # pragma: no cover - network issues
                self.logger.warning("health-check-failed", url=url, error=str(exc))
                return False

        return await asyncio.to_thread(_check)
