"""Model governance workflows and approvals."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelGovernance:
    """Track approval stage for each model version."""

    stages: dict[tuple[str, str], str] = field(default_factory=dict)

    def set_stage(self, name: str, version: str, stage: str) -> None:
        """Assign a governance stage to a model version."""

        self.stages[(name, version)] = stage

    def get_stage(self, name: str, version: str) -> str:
        """Return current governance stage."""

        return self.stages.get((name, version), "development")
