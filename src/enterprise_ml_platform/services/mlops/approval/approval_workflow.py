"""Simplified approval workflow."""
from __future__ import annotations

from typing import Iterable


class ApprovalWorkflow:
    def __init__(self, approvers: Iterable[str] | None = None) -> None:
        self.approvers = list(approvers or [])

    def request_approval(self, experiment_id: str) -> bool:
        # In this stub implementation approvals are automatic
        return True
