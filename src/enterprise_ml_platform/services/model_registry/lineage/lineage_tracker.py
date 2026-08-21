"""Track lineage information for models and datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class LineageRecord(TypedDict):
    """What a model version was built from.

    Attributes:
        parents: ``(name, version)`` of the model versions it derives from.
        datasets: Identifiers of the datasets it was trained on.
    """

    parents: list[tuple[str, str]]
    datasets: list[str]


@dataclass
class LineageTracker:
    """Maintain parent-child relationships between model versions."""

    lineage: dict[tuple[str, str], LineageRecord] = field(default_factory=dict)

    def record(
        self,
        name: str,
        version: str,
        parents: list[tuple[str, str]] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """Store lineage info for ``name``/``version``."""
        self.lineage[(name, version)] = {
            "parents": list(parents or []),
            "datasets": list(datasets or []),
        }

    def get(self, name: str, version: str) -> LineageRecord:
        """Return lineage information for a model version."""
        return self.lineage.get((name, version), {"parents": [], "datasets": []})
