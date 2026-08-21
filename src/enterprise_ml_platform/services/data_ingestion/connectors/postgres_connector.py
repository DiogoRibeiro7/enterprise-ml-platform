"""PostgreSQL data connector for the ingestion service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pyarrow as pa
import structlog

try:  # pragma: no cover - optional dependency
    import asyncpg
except Exception:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore

from .base import AsyncDataConnector


class PostgresDataConnector(AsyncDataConnector):
    """Asynchronous connector for PostgreSQL databases.

    Parameters
    ----------
    dsn:
        Connection string or parameters for :mod:`asyncpg`.
    """

    def __init__(self, dsn: str, **connect_kwargs: Any) -> None:
        self.dsn = dsn
        self.connect_kwargs = connect_kwargs
        self._pool: asyncpg.Pool | None = None
        self._log = structlog.get_logger().bind(connector="postgres")

    async def connect(self) -> None:
        if asyncpg is None:  # pragma: no cover - dependency guard
            raise RuntimeError("asyncpg is required for Postgres connector")
        self._pool = await asyncpg.create_pool(dsn=self.dsn, **self.connect_kwargs)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def read(
        self,
        query: str,
        chunk_size: int = 10000,
    ) -> AsyncIterator[pd.DataFrame]:
        """Execute a query and stream results in chunks."""

        if not self._pool:
            raise RuntimeError("Connector not connected")

        async with self._pool.acquire() as connection:
            statement = connection.cursor(query)
            batch = []
            async for record in statement:
                batch.append(dict(record))
                if len(batch) >= chunk_size:
                    yield pd.DataFrame(batch)
                    batch.clear()
            if batch:
                yield pd.DataFrame(batch)

    async def get_schema(self) -> pa.Schema:
        if not self._pool:
            raise RuntimeError("Connector not connected")
        async with self._pool.acquire() as connection:
            table = await connection.fetch("SELECT 1")
            df = pd.DataFrame([dict(r) for r in table])
            return pa.Table.from_pandas(df).schema
