"""Automated data quality remediation strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataRemediation:
    """Apply simple fixes for common data quality issues."""

    def remediate(self, df: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
        """Return a corrected DataFrame based on ``issues``."""

        result = df.copy()
        if any("missing column" in i for i in issues):
            # nothing we can do automatically here
            pass
        if any("missing values" in i for i in issues):
            result = result.fillna(0)
        return result
