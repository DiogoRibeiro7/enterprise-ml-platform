"""Manage job queues for resource allocation."""

from __future__ import annotations

from collections import deque


class QueueManager:
    """FIFO queue storing jobs and requested resources."""

    def __init__(self) -> None:
        self._queue: deque[tuple[str, dict[str, int]]] = deque()

    def enqueue(self, job_id: str, resources: dict[str, int] | None = None) -> None:
        self._queue.append((job_id, resources or {}))

    def dequeue(self) -> tuple[str, dict[str, int]] | None:
        if self._queue:
            return self._queue.popleft()
        return None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)
