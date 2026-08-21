"""Tests for the offline feature stores.

Both implementations are held to the same contract, so a feature set behaves
identically whether it is served from memory or from Parquet. The persistence
and point-in-time tests are the ones that matter: a training set must never
contain a value recorded after the label it is paired with.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from enterprise_ml_platform.services.feature_store import (
    InMemoryOfflineStore,
    ParquetOfflineStore,
)


@pytest.fixture(params=["memory", "parquet"])
def store(request, tmp_path: pathlib.Path):
    """Each offline store implementation, behind the shared interface."""
    if request.param == "memory":
        return InMemoryOfflineStore()
    return ParquetOfflineStore(tmp_path / "offline")


HISTORY = pd.DataFrame(
    [
        {"entity_id": "42", "score": 0.1, "timestamp": pd.Timestamp("2024-01-01")},
        {"entity_id": "42", "score": 0.5, "timestamp": pd.Timestamp("2024-06-01")},
        {"entity_id": "42", "score": 0.9, "timestamp": pd.Timestamp("2024-12-01")},
        {"entity_id": "43", "score": 0.2, "timestamp": pd.Timestamp("2024-01-01")},
    ]
)


# ----------------------------------------------------------------------
# Shared contract
# ----------------------------------------------------------------------
async def test_round_trips_a_feature_set(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    frame = await store.read_feature_set("customers", "v1")

    assert len(frame) == 4
    assert set(frame.columns) == {"entity_id", "score", "timestamp"}


async def test_latest_value_is_returned_without_a_cutoff(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    assert await store.get_features("customers", "v1", "42") == {"score": 0.9}


async def test_entities_are_isolated(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    assert await store.get_features("customers", "v1", "43") == {"score": 0.2}


async def test_versions_are_isolated(store) -> None:
    await store.write_features("customers", "v1", HISTORY)
    await store.write_features(
        "customers", "v2", pd.DataFrame([{"entity_id": "42", "score": 99.0}])
    )

    assert await store.get_features("customers", "v1", "42") == {"score": 0.9}
    assert await store.get_features("customers", "v2", "42") == {"score": 99.0}


async def test_unknown_feature_set_returns_empty(store) -> None:
    assert await store.get_features("nope", "v1", "42") == {}
    assert (await store.read_feature_set("nope", "v1")).empty


async def test_unknown_entity_returns_empty(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    assert await store.get_features("customers", "v1", "999") == {}


async def test_entity_ids_are_matched_regardless_of_source_dtype(store) -> None:
    """Ids written as integers must still be found when looked up as strings."""
    await store.write_features(
        "customers", "v1", pd.DataFrame([{"entity_id": 42, "score": 0.7}])
    )

    assert await store.get_features("customers", "v1", "42") == {"score": 0.7}


# ----------------------------------------------------------------------
# Point-in-time correctness
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "as_of,expected",
    [
        ("2024-03-01", 0.1),
        ("2024-06-01", 0.5),
        ("2024-07-01", 0.5),
        ("2025-01-01", 0.9),
    ],
)
async def test_point_in_time_lookup_never_sees_the_future(
    store, as_of, expected
) -> None:
    await store.write_features("customers", "v1", HISTORY)

    served = await store.get_features(
        "customers", "v1", "42", as_of=pd.Timestamp(as_of)
    )

    assert served == {"score": expected}


async def test_cutoff_before_any_record_returns_empty(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    served = await store.get_features(
        "customers", "v1", "42", as_of=pd.Timestamp("2023-01-01")
    )

    assert served == {}


async def test_cutoff_is_inclusive_of_its_own_timestamp(store) -> None:
    await store.write_features("customers", "v1", HISTORY)

    served = await store.get_features(
        "customers", "v1", "42", as_of=pd.Timestamp("2024-01-01")
    )

    assert served == {"score": 0.1}


# ----------------------------------------------------------------------
# Persistence: the difference between the two implementations
# ----------------------------------------------------------------------
async def test_parquet_store_survives_a_restart(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "offline"
    await ParquetOfflineStore(root).write_features("customers", "v1", HISTORY)

    # A brand new store object, as a restarted process would build.
    reopened = ParquetOfflineStore(root)

    assert await reopened.get_features("customers", "v1", "42") == {"score": 0.9}


async def test_in_memory_store_does_not_survive_a_restart() -> None:
    """Named for what it is: the data is gone with the process."""
    await InMemoryOfflineStore().write_features("customers", "v1", HISTORY)

    assert await InMemoryOfflineStore().get_features("customers", "v1", "42") == {}


async def test_parquet_files_are_laid_out_predictably(tmp_path: pathlib.Path) -> None:
    store = ParquetOfflineStore(tmp_path / "offline")
    await store.write_features("customers", "v1", HISTORY)

    path = store.path_for("customers", "v1")

    assert path.exists()
    assert path.relative_to(store.root).as_posix() == "customers/v1/data.parquet"


async def test_parquet_versions_are_written_to_separate_files(
    tmp_path: pathlib.Path,
) -> None:
    store = ParquetOfflineStore(tmp_path / "offline")
    await store.write_features("customers", "v1", HISTORY)
    await store.write_features("customers", "v2", HISTORY)

    assert store.path_for("customers", "v1") != store.path_for("customers", "v2")
    assert store.path_for("customers", "v1").exists()
    assert store.path_for("customers", "v2").exists()


# ----------------------------------------------------------------------
# The query is parameterised
# ----------------------------------------------------------------------
async def test_feature_set_name_cannot_escape_into_the_query(
    tmp_path: pathlib.Path,
) -> None:
    """The Parquet path is caller-derived, so it must be bound, not spliced.

    The path is built from the feature set name and version. Interpolating it
    into the SQL text would let a crafted name close the string literal and
    read whatever file it names.
    """
    store = ParquetOfflineStore(tmp_path / "offline")
    await store.write_features("customers", "v1", HISTORY)
    secret = tmp_path / "secret.parquet"
    pd.DataFrame([{"entity_id": "42", "score": 999.0}]).to_parquet(secret, index=False)

    hostile = f"x'), read_parquet('{secret.as_posix()}"

    assert await store.get_features(hostile, "v1", "42") == {}
    assert await store.get_features("customers", hostile, "42") == {}


async def test_entity_id_cannot_escape_into_the_query(tmp_path: pathlib.Path) -> None:
    store = ParquetOfflineStore(tmp_path / "offline")
    await store.write_features("customers", "v1", HISTORY)

    assert await store.get_features("customers", "v1", "42' OR '1'='1") == {}
