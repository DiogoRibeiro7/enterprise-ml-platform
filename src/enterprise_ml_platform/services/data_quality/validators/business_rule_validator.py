"""Validation of custom business rules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd

Rule = Callable[[pd.DataFrame], Iterable[str]]


@dataclass
class BusinessRuleValidator:
    """Apply domain specific validation rules to a dataset."""

    rules: list[Rule] = field(default_factory=list)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Run all configured rules against ``df``."""

        errors: list[str] = []
        for rule in self.rules:
            errors.extend(list(rule(df)))
        return errors
