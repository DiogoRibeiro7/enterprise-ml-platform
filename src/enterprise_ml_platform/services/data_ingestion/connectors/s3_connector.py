"""AWS S3 data connector for the ingestion service."""

from __future__ import annotations

import asyncio
import io
from typing import Any, AsyncIterator, Dict, Optional

import pandas as pd
import pyarrow as pa
import structlog

try:  # pragma: no cover - optional dependency
    import aioboto3
except Exception:  # pragma: no cover - optional dependency
    aioboto3 = None  # type: ignore

from .base import AsyncDataConnector


class S3DataConnector(AsyncDataConnector):
    """Asynchronous connector for reading objects from Amazon S3.

    Parameters
    ----------
    bucket:
        Name of the S3 bucket.
    aws_region:
        Optional region where the bucket resides.
    """

    def __init__(self, bucket: str, aws_region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.aws_region = aws_region
        self._session: Optional[aioboto3.Session] = None
        self._client: Any = None
        self._log = structlog.get_logger().bind(connector="s3", bucket=bucket)

    async def connect(self) -> None:
        if aioboto3 is None:  # pragma: no cover - dependency guard
            raise RuntimeError("aioboto3 is required for S3 connector")
        self._session = aioboto3.Session(region_name=self.aws_region)
        self._client = await self._session.client("s3").__aenter__()

    async def disconnect(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._session = None

    async def read(
        self,
        prefix: str = "",
        batch_size: int = 50,
        max_parallel: int = 10,
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream objects from S3 and yield them as data frames."""

        if not self._client:
            raise RuntimeError("Connector not connected")

        paginator = self._client.get_paginator("list_objects_v2")
        sem = asyncio.Semaphore(max_parallel)
        tasks = []

        async def _fetch(key: str) -> Optional[pd.DataFrame]:
            async with sem:
                try:
                    obj = await self._client.get_object(Bucket=self.bucket, Key=key)
                    data = await obj["Body"].read()
                    if key.endswith(".parquet"):
                        return pd.read_parquet(io.BytesIO(data))
                    if key.endswith(".csv"):
                        return pd.read_csv(io.BytesIO(data))
                    if key.endswith(".json"):
                        return pd.read_json(io.BytesIO(data), lines=True)
                except Exception as exc:  # pragma: no cover
                    self._log.warning("object fetch failed", key=key, error=str(exc))
            return None

        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                tasks.append(asyncio.create_task(_fetch(obj["Key"])))
                if len(tasks) >= batch_size:
                    for df in await asyncio.gather(*tasks):
                        if df is not None:
                            yield df
                    tasks.clear()

        if tasks:
            for df in await asyncio.gather(*tasks):
                if df is not None:
                    yield df

    async def get_schema(self) -> pa.Schema:
        if not self._client:
            raise RuntimeError("Connector not connected")
        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.bucket, MaxKeys=1):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                obj = await self._client.get_object(Bucket=self.bucket, Key=key)
                data = await obj["Body"].read()
                if key.endswith(".parquet"):
                    table = pa.parquet.read_table(io.BytesIO(data))
                else:
                    df = pd.read_csv(io.BytesIO(data))
                    table = pa.Table.from_pandas(df)
                return table.schema
        raise RuntimeError("No objects found to infer schema")
