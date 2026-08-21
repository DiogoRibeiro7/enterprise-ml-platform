"""Minimal Slack notification client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SlackClient:
    """Pretend Slack webhook sender used in tests."""

    webhook: str | None = None

    async def send(self, alert: Any) -> None:  # pragma: no cover - simple logging
        await asyncio.sleep(0)
        logger.info("slack_alert", webhook=self.webhook, message=alert.message)
