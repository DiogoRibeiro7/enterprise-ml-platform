"""Evaluate alerting rules against metric values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Alert:
    """Simple alert dataclass produced by :class:`RulesEngine`."""

    name: str
    severity: str
    message: str


@dataclass
class AlertRule:
    """Rule defining when an alert should fire."""

    metric: str
    threshold: float
    operator: str = "gt"  # gt, lt, eq
    severity: str = "warning"
    message: str | None = None


class RulesEngine:
    """Evaluate metrics against configured alert rules."""

    def __init__(self, rules: Sequence[AlertRule] | None = None) -> None:
        self.rules = list(rules or [])

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def evaluate(self, metrics: dict[str, float]) -> list[Alert]:
        alerts: list[Alert] = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if self._compare(value, rule.threshold, rule.operator):
                msg = (
                    rule.message
                    or f"{rule.metric} {value} {rule.operator} {rule.threshold}"
                )
                alerts.append(Alert(rule.metric, rule.severity, msg))
        return alerts

    @staticmethod
    def _compare(value: float, threshold: float, operator: str) -> bool:
        if operator == "gt":
            return value > threshold
        if operator == "lt":
            return value < threshold
        if operator == "eq":
            return value == threshold
        raise ValueError(f"Unknown operator: {operator}")
