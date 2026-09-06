"""PostgreSQL data connector for the ingestion service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pyarrow as pa
import structlog

try:  # pragma: no cover - optional dependency
    import asyncpg as _asyncpg
except ImportError:  # pragma: no cover - optional dependency
    _asyncpg = None

from .base import AsyncDataConnector


class PostgresDataConnector(AsyncDataConnector):
    """Read PostgreSQL query results through an asynchronous connection pool."""

    def __init__(self, dsn: str, **connect_kwargs: Any) -> None:
        self.dsn = dsn
        self.connect_kwargs = connect_kwargs
        self._pool: Any | None = None
        self._log = structlog.get_logger().bind(connector="postgres")

    async def connect(self) -> None:
        """Create the PostgreSQL connection pool."""
        if _asyncpg is None:  # pragma: no cover - dependency guard
            raise RuntimeError("asyncpg is required for Postgres connector")
        self._pool = await _asyncpg.create_pool(dsn=self.dsn, **self.connect_kwargs)

    async def disconnect(self) -> None:
        """Close the connection pool when it exists."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def read(self, **config: Any) -> AsyncIterator[pd.DataFrame]:
        """Execute ``query`` and stream records in bounded chunks."""
        if self._pool is None:
            raise RuntimeError("Connector not connected")
        query = config.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        chunk_size = int(config.get("chunk_size", 10000))
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        async with self._pool.acquire() as connection:
            records = connection.cursor(query)
            batch: list[dict[str, Any]] = []
            async for record in records:
                batch.append(dict(record))
                if len(batch) >= chunk_size:
                    yield pd.DataFrame.from_records(batch)
                    batch.clear()
            if batch:
                yield pd.DataFrame.from_records(batch)

    async def get_schema(self) -> pa.Schema:
        """Infer a schema from a minimal query against the connection."""
        if self._pool is None:
            raise RuntimeError("Connector not connected")
        async with self._pool.acquire() as connection:
            records = await connection.fetch("SELECT 1")
            frame = pd.DataFrame.from_records([dict(record) for record in records])
            return pa.Table.from_pandas(frame).schema
