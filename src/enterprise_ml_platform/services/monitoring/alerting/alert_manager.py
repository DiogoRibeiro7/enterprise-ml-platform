from __future__ import annotations

"""Alert manager dispatching notifications to various channels."""

from typing import Dict, Iterable

import structlog

from .notification_clients import EmailClient, SlackClient, PagerDutyClient
from .rules_engine import Alert

logger = structlog.get_logger(__name__)


class AlertManager:
    """Dispatch alerts to registered notification clients."""

    def __init__(self, clients: Dict[str, object] | None = None) -> None:
        self.clients = clients or {
            "email": EmailClient(),
            "slack": SlackClient(),
            "pagerduty": PagerDutyClient(),
        }

    async def dispatch(self, alerts: Iterable[Alert]) -> None:
        """Send alerts to all clients."""
        for alert in alerts:
            for client in self.clients.values():
                try:
                    await client.send(alert)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logger.warning("alert_send_failed", error=str(exc))

    async def close(self) -> None:
        """Close any client resources."""
        for client in self.clients.values():
            close = getattr(client, "close", None)
            if close:
                await close()
