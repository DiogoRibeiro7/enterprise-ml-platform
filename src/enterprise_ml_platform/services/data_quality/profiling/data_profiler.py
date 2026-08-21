"""Compute descriptive statistics for datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataProfiler:
    """Generate simple statistical profiles for a DataFrame."""

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return summary statistics for ``df``."""

        summary = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing": df.isna().mean().to_dict(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "stats": df.describe(include="all").to_dict(),
        }
        return summary
