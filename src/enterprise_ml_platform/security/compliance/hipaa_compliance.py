"""HIPAA compliance helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HIPAACompliance:
    """Tracks access to PHI and associated safeguards."""

    phi_access_log: list[str] = field(default_factory=list)
    risk_assessments: dict[str, str] = field(default_factory=dict)

    def log_phi_access(self, user: str, record_id: str) -> None:
        self.phi_access_log.append(f"{user}:{record_id}")

    def record_risk_assessment(self, system: str, status: str) -> None:
        self.risk_assessments[system] = status
