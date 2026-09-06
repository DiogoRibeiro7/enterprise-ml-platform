"""AWS S3 data connector for the ingestion service."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

try:  # pragma: no cover - optional dependency
    import aioboto3 as _aioboto3
except ImportError:  # pragma: no cover - optional dependency
    _aioboto3 = None

from .base import AsyncDataConnector


class S3DataConnector(AsyncDataConnector):
    """Read CSV, JSON Lines, and Parquet objects from Amazon S3."""

    def __init__(self, bucket: str, aws_region: str | None = None) -> None:
        self.bucket = bucket
        self.aws_region = aws_region
        self._session: Any | None = None
        self._client_context: Any | None = None
        self._client: Any | None = None
        self._log = structlog.get_logger().bind(connector="s3", bucket=bucket)

    async def connect(self) -> None:
        """Open an asynchronous S3 client."""
        if _aioboto3 is None:  # pragma: no cover - dependency guard
            raise RuntimeError("aioboto3 is required for S3 connector")
        self._session = _aioboto3.Session(region_name=self.aws_region)
        self._client_context = self._session.client("s3")
        self._client = await self._client_context.__aenter__()

    async def disconnect(self) -> None:
        """Close the S3 client context when it exists."""
        if self._client_context is not None:
            await self._client_context.__aexit__(None, None, None)
        self._client = None
        self._client_context = None
        self._session = None

    async def read(self, **config: Any) -> AsyncIterator[pd.DataFrame]:
        """Stream supported objects under ``prefix`` as data frames."""
        if self._client is None:
            raise RuntimeError("Connector not connected")
        client = self._client
        prefix = str(config.get("prefix", ""))
        batch_size = int(config.get("batch_size", 50))
        max_parallel = int(config.get("max_parallel", 10))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")

        paginator = client.get_paginator("list_objects_v2")
        semaphore = asyncio.Semaphore(max_parallel)
        tasks: list[asyncio.Task[pd.DataFrame | None]] = []

        async def fetch(key: str) -> pd.DataFrame | None:
            async with semaphore:
                try:
                    response = await client.get_object(Bucket=self.bucket, Key=key)
                    data = await response["Body"].read()
                    if key.endswith(".parquet"):
                        return pd.read_parquet(io.BytesIO(data))
                    if key.endswith(".csv"):
                        return pd.read_csv(io.BytesIO(data))
                    if key.endswith(".json"):
                        return pd.read_json(io.BytesIO(data), lines=True)
                except Exception as exc:  # pragma: no cover - remote operation
                    self._log.warning("object fetch failed", key=key, error=str(exc))
            return None

        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                tasks.append(asyncio.create_task(fetch(str(item["Key"]))))
                if len(tasks) >= batch_size:
                    for frame in await asyncio.gather(*tasks):
                        if frame is not None:
                            yield frame
                    tasks.clear()

        if tasks:
            for frame in await asyncio.gather(*tasks):
                if frame is not None:
                    yield frame

    async def get_schema(self) -> pa.Schema:
        """Infer an Arrow schema from the first object in the bucket."""
        if self._client is None:
            raise RuntimeError("Connector not connected")
        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.bucket, MaxKeys=1):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                response = await self._client.get_object(Bucket=self.bucket, Key=key)
                data = await response["Body"].read()
                if key.endswith(".parquet"):
                    table = pq.read_table(io.BytesIO(data))
                else:
                    frame = pd.read_csv(io.BytesIO(data))
                    table = pa.Table.from_pandas(frame)
                return table.schema
        raise RuntimeError("No objects found to infer schema")
