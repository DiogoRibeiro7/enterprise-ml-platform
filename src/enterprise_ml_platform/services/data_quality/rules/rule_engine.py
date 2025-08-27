"""Lightweight validation rule engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List

import pandas as pd

Rule = Callable[[pd.DataFrame], Iterable[str]]


@dataclass
class RuleEngine:
    """Manage validation rules and execute them."""

    rules: Dict[str, Rule] = field(default_factory=dict)

    def add_rule(self, name: str, rule: Rule) -> None:
        self.rules[name] = rule

    def run(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Execute all rules against ``df``."""

        results: Dict[str, List[str]] = {}
        for name, rule in self.rules.items():
            results[name] = list(rule(df))
        return results
