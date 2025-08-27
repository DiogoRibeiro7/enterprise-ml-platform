"""Simple case management workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CaseManagement:
    cases: List[Dict] = field(default_factory=list)

    def create_case(self, alert: Dict) -> Dict:
        case = {"case_id": len(self.cases) + 1, "alert": alert, "status": "open"}
        self.cases.append(case)
        return case

    def resolve_case(self, case_id: int, outcome: str) -> None:
        for case in self.cases:
            if case["case_id"] == case_id:
                case["status"] = outcome
                break

    def open_cases(self) -> List[Dict]:
        return [c for c in self.cases if c["status"] == "open"]
