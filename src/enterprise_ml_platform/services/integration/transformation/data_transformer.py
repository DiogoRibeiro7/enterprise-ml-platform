"""Utilities for request and response transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DataTransformer:
    """Performs lightweight transformation steps.

    Real deployments might integrate full ETL/ELT logic, schema validation or
    protobuf/JSON conversion.  The placeholder simply passes data through so the
    gateway can be exercised in tests.
    """

    def transform_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return request

    def transform_response(self, response: Any) -> Any:
        return response
