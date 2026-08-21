"""Logging configuration utilities for the Enterprise ML Platform.

This module configures `structlog` to output JSON-formatted logs with a
correlation identifier. Two handlers are configured by default: a
``StreamHandler`` for console output and an optional ``FileHandler`` when a
log file path is provided.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

import structlog
from structlog.typing import Processor

# Context variable used to store correlation IDs
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for subsequent log records.

    Args:
        correlation_id: Identifier that ties related log entries together.
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str:
    """Retrieve the current correlation ID.

    Returns:
        The correlation identifier for the current context.
    """
    return _correlation_id.get()


def configure_logging(
    *, log_level: int = logging.INFO, log_file: str | None = None
) -> None:
    """Configure application-wide structured logging.

    Args:
        log_level: Logging level for all handlers.
        log_file: Optional file path for a dedicated ``FileHandler``.
    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, handlers=handlers, format="%(message)s")
