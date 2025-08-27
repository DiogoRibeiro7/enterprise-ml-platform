"""Statistical testing utilities for experiments."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Run significance tests and compute effect sizes."""

    def t_test(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        effect = np.mean(b) - np.mean(a)
        return {"t_stat": float(t_stat), "p_value": float(p_val), "effect": float(effect)}

    def chi_square(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        table = np.array([a, b])
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        return {"chi2": float(chi2), "p_value": float(p_val)}

    def bayesian(self, a_success: int, a_fail: int, b_success: int, b_fail: int) -> float:
        a_samples = np.random.beta(a_success + 1, a_fail + 1, 10000)
        b_samples = np.random.beta(b_success + 1, b_fail + 1, 10000)
        return float((b_samples > a_samples).mean())

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if len(data["variants"]) != 2:
            return {"detail": "only two-variant analysis supported"}
        vnames = sorted(data["variants"].keys())
        a = np.array(data["variants"][vnames[0]]["values"])
        b = np.array(data["variants"][vnames[1]]["values"])
        t_res = self.t_test(a, b)
        conv_a = np.array(data["variants"][vnames[0]]["success_fail"])
        conv_b = np.array(data["variants"][vnames[1]]["success_fail"])
        chi_res = self.chi_square(conv_a, conv_b)
        bayes = self.bayesian(
            int(conv_a[0]), int(conv_a[1]), int(conv_b[0]), int(conv_b[1])
        )
        return {"t_test": t_res, "chi_square": chi_res, "bayesian_prob": bayes}
