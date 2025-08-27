"""Health check schemas."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Represents the health of the application."""

    status: str
    details: Optional[Dict[str, str]] = None
