from __future__ import annotations

"""Model governance workflows and approvals."""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ModelGovernance:
    """Track approval stage for each model version."""

    stages: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def set_stage(self, name: str, version: str, stage: str) -> None:
        """Assign a governance stage to a model version."""

        self.stages[(name, version)] = stage

    def get_stage(self, name: str, version: str) -> str:
        """Return current governance stage."""

        return self.stages.get((name, version), "development")
