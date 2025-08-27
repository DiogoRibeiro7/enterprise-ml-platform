"""Real-time model inference utilities."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import structlog

logger = structlog.get_logger()


class StreamPredictor:
    """Perform low-latency predictions on streaming features."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.logger = logger.bind(component="stream-predictor")

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return prediction for given ``features``.

        This default implementation offloads prediction to a thread to avoid
        blocking the event loop.
        """

        self.logger.debug("predict", features=features)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.model.predict, features)
        return {"value": result}
