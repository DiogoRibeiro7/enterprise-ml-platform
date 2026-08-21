"""Checkpoint management for streaming pipelines."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class CheckpointManager:
    """Persist offsets for fault tolerance and recovery."""

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = str(Path(tempfile.gettempdir()) / "stream.checkpoint")
        self.path = Path(path)
        self.logger = logger.bind(component="checkpoint-manager")

    async def mark_checkpoint(self, message: dict[str, Any]) -> None:
        """Persist checkpoint metadata to disk."""
        try:
            with self.path.open("w") as f:
                json.dump(message, f)
        except Exception as exc:  # pragma: no cover - filesystem errors
            self.logger.warning("checkpoint-failed", error=str(exc))

    async def close(self) -> None:
        self.logger.info("checkpoint-closed")
