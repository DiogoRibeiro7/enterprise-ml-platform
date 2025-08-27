from __future__ import annotations

"""Automated response manager for monitoring alerts."""

from typing import Awaitable, Callable, Iterable

import structlog

from ..alerting.rules_engine import Alert

logger = structlog.get_logger(__name__)


class AutomatedResponder:
    """Trigger actions such as retraining or rollback based on alerts."""

    def __init__(
        self,
        retrain: Callable[[Alert], Awaitable[None]] | None = None,
        rollback: Callable[[Alert], Awaitable[None]] | None = None,
    ) -> None:
        self.retrain = retrain or (lambda alert: logger.info("retrain_trigger", alert=alert))
        self.rollback = rollback or (lambda alert: logger.info("rollback_trigger", alert=alert))

    async def handle(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            if alert.severity == "critical" and "performance" in alert.name:
                await self.rollback(alert)
            elif alert.severity in {"warning", "critical"}:
                await self.retrain(alert)
