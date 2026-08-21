"""Lightweight validation rule engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd

Rule = Callable[[pd.DataFrame], Iterable[str]]


@dataclass
class RuleEngine:
    """Manage validation rules and execute them."""

    rules: dict[str, Rule] = field(default_factory=dict)

    def add_rule(self, name: str, rule: Rule) -> None:
        self.rules[name] = rule

    def run(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Execute all rules against ``df``."""

        results: dict[str, list[str]] = {}
        for name, rule in self.rules.items():
            results[name] = list(rule(df))
        return results
