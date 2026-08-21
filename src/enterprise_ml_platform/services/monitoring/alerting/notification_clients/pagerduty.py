"""Minimal PagerDuty notification client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PagerDutyClient:
    """Pretend PagerDuty client used in tests."""

    routing_key: str | None = None

    async def send(self, alert: Any) -> None:  # pragma: no cover - simple logging
        await asyncio.sleep(0)
        logger.info(
            "pagerduty_alert", routing_key=self.routing_key, message=alert.message
        )
