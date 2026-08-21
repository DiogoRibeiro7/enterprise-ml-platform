"""Simple schema validator using pandas dtypes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureSchemaValidator:
    """Validate pandas DataFrames against an expected schema."""

    expected_schema: dict[str, str]

    def validate(self, df: pd.DataFrame) -> None:
        missing = set(self.expected_schema) - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        for col, dtype in self.expected_schema.items():
            if str(df[col].dtype) != dtype:
                raise ValueError(
                    f"column '{col}' has dtype {df[col].dtype} expected {dtype}"
                )
