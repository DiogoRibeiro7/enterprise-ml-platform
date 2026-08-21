"""Simple task scheduling utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover - avoids a circular import at runtime
    from ..load_balancing.load_balancer import LoadBalancer

logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Very small task scheduler used by :class:`ComputeEngine`.

    The scheduler delegates the actual execution to a cluster manager chosen by
    the load balancer.  The implementation is intentionally tiny but provides a
    single place where more sophisticated policies could be plugged in.
    """

    def __init__(self, load_balancer: LoadBalancer) -> None:
        self.load_balancer = load_balancer

    def schedule(
        self, fn: Callable[..., Any], framework: str, *args: Any, **kwargs: Any
    ) -> Any:
        logger.debug("scheduling task", framework=framework)
        manager = self.load_balancer.choose_manager(framework)
        return manager.submit(fn, *args, **kwargs)
