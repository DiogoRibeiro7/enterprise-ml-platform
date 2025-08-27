"""Schema validation utilities for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class SchemaValidator:
    """Validate datasets against an expected schema.

    The validator performs basic checks around column presence and
    dtypes. It is intentionally lightweight but provides hooks for
    integrating with an external schema registry if required.
    """

    schema: Dict[str, str]

    def validate(self, df: pd.DataFrame) -> List[str]:
        """Validate ``df`` against the configured ``schema``.

        Args:
            df: DataFrame to validate.

        Returns:
            A list of human readable validation error messages. Empty
            list indicates the data conforms to the schema.
        """

        errors: List[str] = []
        for column, expected_dtype in self.schema.items():
            if column not in df.columns:
                errors.append(f"missing column: {column}")
                continue
            actual = str(df[column].dtype)
            if expected_dtype and actual != expected_dtype:
                errors.append(
                    f"column {column} has dtype {actual}, expected {expected_dtype}"
                )
        return errors

    @classmethod
    def infer_schema(cls, df: pd.DataFrame) -> Dict[str, str]:
        """Infer a simple schema mapping from ``df``."""

        return {col: str(dtype) for col, dtype in df.dtypes.items()}
