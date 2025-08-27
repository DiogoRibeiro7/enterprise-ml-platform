"""Minimal helpers for demonstrating GDPR compliance workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GDPRCompliance:
    """Tracks user consent and deletion requests.

    This is intentionally lightweight; real systems would integrate with
    dedicated compliance tooling and legal workflows.
    """

    consent_registry: Dict[str, bool] = field(default_factory=dict)
    deletion_requests: Dict[str, bool] = field(default_factory=dict)

    def record_consent(self, user_id: str, granted: bool) -> None:
        self.consent_registry[user_id] = granted

    def has_consent(self, user_id: str) -> bool:
        return self.consent_registry.get(user_id, False)

    def request_deletion(self, user_id: str) -> None:
        self.deletion_requests[user_id] = True

    def should_delete(self, user_id: str) -> bool:
        return self.deletion_requests.get(user_id, False)

