"""Manage spot instance lifecycle."""

from __future__ import annotations


class SpotManager:
    """Simple spot instance allocator stub."""

    def __init__(self) -> None:
        self.active: dict[str, str] = {}

    def request_instance(self, job_id: str) -> str:
        """Allocate a spot instance for a job."""
        instance_id = f"spot-{job_id}"
        self.active[job_id] = instance_id
        return instance_id

    def release_instance(self, job_id: str) -> None:
        self.active.pop(job_id, None)
