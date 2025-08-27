from __future__ import annotations

"""Manage job queues for resource allocation."""

from collections import deque
from typing import Deque, Dict, Tuple


class QueueManager:
    """FIFO queue storing jobs and requested resources."""

    def __init__(self) -> None:
        self._queue: Deque[Tuple[str, Dict[str, int]]] = deque()

    def enqueue(self, job_id: str, resources: Dict[str, int] | None = None) -> None:
        self._queue.append((job_id, resources or {}))

    def dequeue(self) -> Tuple[str, Dict[str, int]] | None:
        if self._queue:
            return self._queue.popleft()
        return None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)
