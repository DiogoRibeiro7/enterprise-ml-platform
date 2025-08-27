"""Interface with the model deployment service."""
from __future__ import annotations

from typing import Optional


class DeploymentIntegration:
    """Placeholder hooks for deployment actions."""

    def promote(self, model_id: str) -> None:  # pragma: no cover - illustrative
        print(f"Promoting model {model_id}")

    def rollback(self, model_id: str, previous: Optional[str] = None) -> None:  # pragma: no cover
        print(f"Rolling back model {model_id} to {previous}")
