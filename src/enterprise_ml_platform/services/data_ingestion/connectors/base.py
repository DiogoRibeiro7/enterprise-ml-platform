"""Base definitions for asynchronous data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

import pandas as pd
import pyarrow as pa


class AsyncDataConnector(ABC):
    """Abstract base class for asynchronous data source connectors.

    Implementations handle the mechanics of connecting to external
    systems and streaming data back as :class:`pandas.DataFrame` objects.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the underlying connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close any open connections and cleanup resources."""

    @abstractmethod
    async def read(self, **config: Dict[str, Any]) -> AsyncIterator[pd.DataFrame]:
        """Stream data from the source.

        Args:
            **config: Connector specific read options.

        Yields:
            Data frames containing the retrieved records.
        """

    @abstractmethod
    async def get_schema(self) -> pa.Schema:
        """Return the schema for data provided by this connector."""
