from __future__ import annotations

"""Deployment health monitoring utilities."""

import asyncio
from urllib import request

import structlog

logger = structlog.get_logger()


class DeploymentHealthChecker:
    """Simple HTTP-based health checker for deployed endpoints."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self.logger = logger.bind(component="health-checker")

    async def check(self, url: str) -> bool:
        """Return ``True`` if ``url`` responds with status < 400."""

        def _check() -> bool:
            try:
                with request.urlopen(url, timeout=self.timeout) as resp:
                    return resp.status < 400
            except Exception as exc:  # pragma: no cover - network issues
                self.logger.warning("health-check-failed", url=url, error=str(exc))
                return False

        return await asyncio.to_thread(_check)
