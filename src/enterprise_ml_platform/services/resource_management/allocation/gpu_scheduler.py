"""Allocate and release GPU resources."""

from __future__ import annotations


class GPUScheduler:
    """Very small GPU allocation manager."""

    def __init__(self, total_gpus: int) -> None:
        self.total = total_gpus
        self.allocated: dict[str, int] = {}

    def available(self) -> int:
        return self.total - sum(self.allocated.values())

    def allocate(self, job_id: str, num: int) -> bool:
        if num <= self.available():
            self.allocated[job_id] = num
            return True
        return False

    def release(self, job_id: str) -> None:
        self.allocated.pop(job_id, None)
