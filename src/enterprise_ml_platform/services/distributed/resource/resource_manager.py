"""Very small in-memory resource tracker."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ResourceUsage:
    cpu: int = 0
    memory: int = 0  # in MB
    gpu: int = 0


class ResourceManager:
    """Track allocated resources for educational purposes."""

    def __init__(self) -> None:
        self.usage = ResourceUsage()

    def allocate(self, cpu: int = 0, memory: int = 0, gpu: int = 0) -> None:
        logger.debug("allocating resources", cpu=cpu, memory=memory, gpu=gpu)
        self.usage.cpu += cpu
        self.usage.memory += memory
        self.usage.gpu += gpu

    def release(self, cpu: int = 0, memory: int = 0, gpu: int = 0) -> None:
        logger.debug("releasing resources", cpu=cpu, memory=memory, gpu=gpu)
        self.usage.cpu -= cpu
        self.usage.memory -= memory
        self.usage.gpu -= gpu

    def snapshot(self) -> dict[str, int]:
        return {
            "cpu": self.usage.cpu,
            "memory": self.usage.memory,
            "gpu": self.usage.gpu,
        }
