"""Validation of custom business rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List

import pandas as pd

Rule = Callable[[pd.DataFrame], Iterable[str]]


@dataclass
class BusinessRuleValidator:
    """Apply domain specific validation rules to a dataset."""

    rules: List[Rule] = field(default_factory=list)

    def validate(self, df: pd.DataFrame) -> List[str]:
        """Run all configured rules against ``df``."""

        errors: List[str] = []
        for rule in self.rules:
            errors.extend(list(rule(df)))
        return errors
