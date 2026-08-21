"""Base definitions for asynchronous data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

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
    def read(self, **config: Any) -> AsyncIterator[pd.DataFrame]:
        """Stream data from the source.

        Implementations are async generators, so calling this returns the
        iterator directly rather than a coroutine that yields one.


        Args:
            **config: Connector specific read options.

        Yields:
            Data frames containing the retrieved records.
        """

    @abstractmethod
    async def get_schema(self) -> pa.Schema:
        """Return the schema for data provided by this connector."""
