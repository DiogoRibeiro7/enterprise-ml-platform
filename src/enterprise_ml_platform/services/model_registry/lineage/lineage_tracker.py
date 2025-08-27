from __future__ import annotations

"""Track lineage information for models and datasets."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LineageTracker:
    """Maintain parent-child relationships between model versions."""

    lineage: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str]]]] = field(
        default_factory=dict
    )

    def record(
        self,
        name: str,
        version: str,
        parents: List[Tuple[str, str]] | None = None,
        datasets: List[str] | None = None,
    ) -> None:
        """Store lineage info for ``name``/``version``."""

        self.lineage[(name, version)] = {
            "parents": parents or [],
            "datasets": datasets or [],
        }

    def get(self, name: str, version: str) -> Dict[str, List[Tuple[str, str]]]:
        """Return lineage information for a model version."""

        return self.lineage.get((name, version), {"parents": [], "datasets": []})
