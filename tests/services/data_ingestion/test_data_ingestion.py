from __future__ import annotations

import io
from collections.abc import AsyncIterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from enterprise_ml_platform.services.data_ingestion import (
    AsyncDataConnector,
    DataIngestionService,
    DataSource,
    DataValidator,
    S3DataConnector,
)
from enterprise_ml_platform.services.data_ingestion import service as service_module


class FakeConnector(AsyncDataConnector):
    def __init__(self, batches: list[pd.DataFrame]) -> None:
        self.batches = batches
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def read(self, **config: object) -> AsyncIterator[pd.DataFrame]:
        for batch in self.batches:
            yield batch.copy()

    async def get_schema(self) -> pa.Schema:
        return pa.Table.from_pandas(self.batches[0]).schema


@pytest.mark.asyncio
async def test_ingest_validates_batches_and_records_metrics() -> None:
    connector = FakeConnector(
        [
            pd.DataFrame({"value": [1.0, None, 1.0]}),
            pd.DataFrame({"value": [2.0]}),
        ]
    )
    service = DataIngestionService()
    service.register_source(
        DataSource(
            name="measurements",
            type="s3",
            connection={},
            quality_rules=[
                {
                    "type": "completeness",
                    "threshold": 1.0,
                    "impute": True,
                    "impute_value": 0.0,
                },
                {"type": "uniqueness"},
            ],
        ),
        connector=connector,
    )

    batches = [batch async for batch in service.ingest("measurements")]

    assert connector.connected
    assert connector.disconnected
    assert [len(batch) for batch in batches] == [2, 1]
    assert not batches[0].isna().any().any()
    assert service.metrics.batches == 2
    assert service.metrics.records_ingested == 4
    assert service.metrics.bytes_processed > 0
    assert service.metrics.ended_at >= service.metrics.started_at


@pytest.mark.asyncio
async def test_validator_exposes_failed_schema_report() -> None:
    validator = DataValidator({"value": "int64"})

    await validator.validate(pd.DataFrame({"value": ["not-an-integer"]}))

    assert validator.last_report is not None
    assert not validator.last_report.passed
    assert validator.last_report.stats["errors"] == 1
    assert "expected int64" in validator.last_report.errors[0]


@pytest.mark.asyncio
async def test_cache_factory_is_not_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    cache = FakeCache()
    monkeypatch.setattr(service_module.aioredis, "from_url", lambda _url: cache)
    service = DataIngestionService(
        {"enabled": True, "redis_url": "redis://cache:6379/0"}
    )

    await service.initialize()
    assert service._cache is cache

    await service.shutdown()
    assert cache.closed


def test_cache_key_is_independent_of_mapping_order() -> None:
    service = DataIngestionService()

    first = service._cache_key("events", {"prefix": "2026/", "limit": 50})
    reordered = service._cache_key("events", {"limit": 50, "prefix": "2026/"})

    assert first == reordered


@pytest.mark.asyncio
async def test_s3_parquet_schema_uses_pyarrow_parquet_reader() -> None:
    payload = io.BytesIO()
    pq.write_table(pa.table({"value": [1, 2]}), payload)

    class Body:
        async def read(self) -> bytes:
            return payload.getvalue()

    class Paginator:
        async def paginate(self, **config: object) -> AsyncIterator[dict[str, object]]:
            yield {"Contents": [{"Key": "sample.parquet"}]}

    class Client:
        def get_paginator(self, operation: str) -> Paginator:
            assert operation == "list_objects_v2"
            return Paginator()

        async def get_object(self, **config: object) -> dict[str, Body]:
            return {"Body": Body()}

    connector = S3DataConnector("test-bucket")
    connector._client = Client()

    schema = await connector.get_schema()

    assert schema.names == ["value"]
