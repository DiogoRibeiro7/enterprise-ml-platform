"""Custom metrics shared across the platform."""

from __future__ import annotations

from prometheus_client import Gauge

# Example custom metric used by the training service
PENDING_JOBS = Gauge("ml_pending_jobs", "Number of pending training jobs")
