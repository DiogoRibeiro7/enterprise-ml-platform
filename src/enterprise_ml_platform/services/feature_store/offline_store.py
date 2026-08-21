"""Offline feature storage.

The offline store holds the full history of a feature set. It answers two
questions the online store cannot: what did this entity look like *at a given
point in time* (so a training set never contains values recorded after its
label), and what is the complete feature set (so a model can be retrained).

Two implementations are provided:

:class:`InMemoryOfflineStore`
    Everything lives in a dictionary. Useful for tests and examples; the data
    is gone when the process exits.
:class:`ParquetOfflineStore`
    Parquet files on disk or object storage, queried with DuckDB. Survives
    restarts and is the one to use for anything real.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..monitoring.collectors.metrics_collector import MetricsCollector

ENTITY_COLUMN = "entity_id"
TIMESTAMP_COLUMN = "timestamp"


def _normalise_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose entity ids are strings.

    Entity ids arrive as whatever dtype the source used. Storing them as
    strings keeps lookups from silently missing when an id written as ``42``
    is later looked up as ``"42"``.
    """
    if ENTITY_COLUMN not in df.columns:
        return df.copy()
    out = df.copy()
    out[ENTITY_COLUMN] = out[ENTITY_COLUMN].astype(str)
    return out


def _row_to_features(row: pd.Series) -> dict[str, float]:
    """Return the row's feature columns, dropping the identifier columns."""
    stripped = row.drop(labels=[ENTITY_COLUMN, TIMESTAMP_COLUMN], errors="ignore")
    return {str(name): value for name, value in stripped.to_dict().items()}


class OfflineStore(ABC):
    """Interface every offline store implements."""

    @abstractmethod
    async def write_features(self, name: str, version: str, df: pd.DataFrame) -> None:
        """Persist ``df`` as version ``version`` of feature set ``name``."""

    @abstractmethod
    async def get_features(
        self,
        name: str,
        version: str,
        entity_id: str,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Return one entity's features, optionally as they stood at ``as_of``."""

    @abstractmethod
    async def read_feature_set(self, name: str, version: str) -> pd.DataFrame:
        """Return the whole feature set, or an empty frame if unknown."""


class InMemoryOfflineStore(OfflineStore):
    """Offline store backed by a dictionary. Data does not survive the process."""

    def __init__(self, metrics: MetricsCollector | None = None) -> None:
        """Create an empty store."""
        self._store: dict[tuple, pd.DataFrame] = {}
        self.metrics = metrics

    # ------------------------------------------------------------------
    async def write_features(self, name: str, version: str, df: pd.DataFrame) -> None:
        """Persist ``df`` as version ``version`` of feature set ``name``."""
        self._store[(name, version)] = _normalise_entities(df)

    # ------------------------------------------------------------------
    async def read_feature_set(self, name: str, version: str) -> pd.DataFrame:
        """Return the whole feature set, or an empty frame if unknown."""
        df = self._store.get((name, version))
        return df.copy() if df is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    async def get_features(
        self,
        name: str,
        version: str,
        entity_id: str,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Return one entity's features, optionally as they stood at ``as_of``."""
        df = self._store.get((name, version))
        if df is None:
            return {}
        rows = df[df[ENTITY_COLUMN] == str(entity_id)]
        if rows.empty:
            return {}
        if TIMESTAMP_COLUMN in rows.columns:
            if as_of is not None:
                rows = rows[rows[TIMESTAMP_COLUMN] <= as_of]
                if rows.empty:
                    return {}
            rows = rows.sort_values(TIMESTAMP_COLUMN).iloc[-1:]
        return _row_to_features(rows.iloc[0])


class ParquetOfflineStore(OfflineStore):
    """Offline store backed by Parquet files and queried with DuckDB.

    Each version is written once to ``{root}/{feature_set}/{version}/data.parquet``
    and never mutated, so a training run can always reproduce the exact inputs
    a model was fitted on.
    """

    def __init__(
        self,
        root: Path | str,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Create a store rooted at ``root``, creating the directory if needed.

        Raises:
            RuntimeError: If DuckDB is not installed.
        """
        try:
            import duckdb  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "duckdb is required for ParquetOfflineStore; "
                "install enterprise-ml-platform[feature-store]"
            ) from exc
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics

    # ------------------------------------------------------------------
    def path_for(self, name: str, version: str) -> Path:
        """Return the Parquet file holding one version of a feature set."""
        return self.root / name / version / "data.parquet"

    # ------------------------------------------------------------------
    async def write_features(self, name: str, version: str, df: pd.DataFrame) -> None:
        """Persist ``df`` as version ``version`` of feature set ``name``."""
        path = self.path_for(name, version)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            _normalise_entities(df).to_parquet(path, index=False)

        await asyncio.to_thread(_write)

    # ------------------------------------------------------------------
    async def read_feature_set(self, name: str, version: str) -> pd.DataFrame:
        """Return the whole feature set, or an empty frame if unknown."""
        path = self.path_for(name, version)
        if not path.exists():
            return pd.DataFrame()
        return await asyncio.to_thread(pd.read_parquet, path)

    # ------------------------------------------------------------------
    async def get_features(
        self,
        name: str,
        version: str,
        entity_id: str,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Return one entity's features, optionally as they stood at ``as_of``."""
        path = self.path_for(name, version)
        if not path.exists():
            return {}
        return await asyncio.to_thread(self._query, path, str(entity_id), as_of)

    # ------------------------------------------------------------------
    @staticmethod
    def _query(
        path: Path, entity_id: str, as_of: pd.Timestamp | None
    ) -> dict[str, float]:
        """Run the point-in-time lookup. Blocking; call via a worker thread.

        The file path is bound as a query parameter rather than interpolated:
        it is derived from a caller-supplied feature set name and version, and
        splicing that into SQL would let a crafted name read arbitrary files.
        """
        import duckdb

        source = str(path).replace("\\", "/")
        with duckdb.connect() as conn:
            columns = {
                row[0]
                for row in conn.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [source]
                ).fetchall()
            }
            # Only the column names are interpolated, and they are module
            # constants -- SQL has no parameter form for identifiers. Every
            # caller-supplied value is bound. # nosec B608
            sql = f"SELECT * FROM read_parquet(?) WHERE {ENTITY_COLUMN} = ?"  # nosec B608
            params: list = [source, entity_id]
            if TIMESTAMP_COLUMN in columns:
                if as_of is not None:
                    sql += f" AND {TIMESTAMP_COLUMN} <= ?"
                    params.append(pd.Timestamp(as_of).to_pydatetime())
                # Newest row that satisfies the cutoff wins.
                sql += f" ORDER BY {TIMESTAMP_COLUMN} DESC"
            sql += " LIMIT 1"
            frame = conn.execute(sql, params).fetch_df()

        if frame.empty:
            return {}
        return _row_to_features(frame.iloc[0])


#: Backwards-compatible alias. The original name did not say that the data
#: only lived for the lifetime of the process.
OfflineFeatureStore = InMemoryOfflineStore
