"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Represents the health of the application."""

    status: str
    details: dict[str, str] | None = None
