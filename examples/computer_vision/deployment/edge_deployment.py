"""Edge device deployment helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EdgeDeployment:
    """Simulates model optimisation and deployment to edge devices."""

    model: Any
    quantized: bool = False

    def quantize(self) -> None:
        """Perform a dummy quantisation step."""
        self.quantized = True

    def deploy(self, device: str) -> str:
        """"Deploy" the model to ``device``."""
        state = "quantized" if self.quantized else "raw"
        return f"deployed {state} model to {device}"
