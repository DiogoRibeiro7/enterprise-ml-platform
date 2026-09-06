"""Base definitions for asynchronous data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pyarrow as pa


class AsyncDataConnector(ABC):
    """Abstract interface implemented by every ingestion connector."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the underlying connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close open connections and release resources."""

    @abstractmethod
    def read(self, **config: Any) -> AsyncIterator[pd.DataFrame]:
        """Return an asynchronous stream of data frames."""

    @abstractmethod
    async def get_schema(self) -> pa.Schema:
        """Return the Arrow schema produced by this connector."""
