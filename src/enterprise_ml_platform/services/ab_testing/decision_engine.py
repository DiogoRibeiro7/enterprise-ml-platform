"""Automated decision making for experiments."""
from __future__ import annotations

from typing import Dict, Any


class DecisionEngine:
    """Select winners and apply stopping rules."""

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

    def decide(self, experiment: str, analysis: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        threshold = criteria.get("p_value", 0.05)
        t_p = analysis["t_test"]["p_value"]
        chi_p = analysis["chi_square"]["p_value"]
        bayes = analysis["bayesian_prob"]
        winner = None
        if t_p < threshold and chi_p < threshold and bayes > 0.5:
            winner = criteria.get("prefer", "b")
        return {"winner": winner, "t_p": t_p, "chi_p": chi_p, "bayesian_prob": bayes}
