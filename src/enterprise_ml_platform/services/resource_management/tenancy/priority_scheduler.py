"""Priority-based job scheduling."""

from __future__ import annotations

from collections import defaultdict, deque


class PriorityScheduler:
    def __init__(self) -> None:
        self.queues: dict[int, deque[str]] = defaultdict(deque)

    def submit(self, job_id: str, priority: int) -> None:
        self.queues[priority].append(job_id)

    def pop(self) -> str | None:
        for priority in sorted(self.queues.keys(), reverse=True):
            queue = self.queues[priority]
            if queue:
                return queue.popleft()
        return None
