from __future__ import annotations

"""Simple resource manager for distributed optimisation."""

from dataclasses import dataclass
import os

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class ResourceManager:
    """Reports basic CPU/GPU availability for schedulers."""

    def available_resources(self) -> dict:
        cpus = os.cpu_count() or 1
        gpus = torch.cuda.device_count() if torch else 0
        return {"cpus": cpus, "gpus": gpus}
