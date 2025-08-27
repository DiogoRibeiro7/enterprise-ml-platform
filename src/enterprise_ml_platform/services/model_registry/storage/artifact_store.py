from __future__ import annotations

"""Minimal model artifact storage."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class ArtifactStore:
    """In-memory artifact registry."""

    artifacts: Dict[Tuple[str, str], Any] = field(default_factory=dict)

    def save(self, name: str, version: str, artifact: Any) -> None:
        """Persist a model artifact."""

        self.artifacts[(name, version)] = artifact

    def get(self, name: str, version: str) -> Any:
        """Retrieve a stored artifact."""

        return self.artifacts.get((name, version))
