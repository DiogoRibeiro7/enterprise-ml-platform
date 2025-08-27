"""Business rule evaluation for fraud detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


Rule = Callable[[Dict], bool]


@dataclass
class RuleEngine:
    """Evaluate configurable rules against transactions."""

    rules: Dict[str, Rule] = field(default_factory=dict)

    def add_rule(self, name: str, rule: Rule) -> None:
        self.rules[name] = rule

    def evaluate(self, txn: Dict) -> List[str]:
        """Return list of rule names triggered for ``txn``."""
        return [name for name, rule in self.rules.items() if rule(txn)]
