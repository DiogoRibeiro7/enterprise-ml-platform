from __future__ import annotations

"""Minimal email notification client."""

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EmailClient:
    """Pretend email sender used in tests."""

    address: str | None = None

    async def send(self, alert: Any) -> None:  # pragma: no cover - simple logging
        await asyncio.sleep(0)
        logger.info("email_alert", address=self.address, message=alert.message)
