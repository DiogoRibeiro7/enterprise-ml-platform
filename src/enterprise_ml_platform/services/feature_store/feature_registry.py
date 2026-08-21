"""Simple in-memory feature registry with optional JSON persistence."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FeatureDescriptor:
    """Metadata describing a registered feature set."""

    name: str
    version: str
    created_at: dt.datetime
    schema: dict[str, str]
    lineage: dict[str, str] | None = None


class FeatureRegistry:
    """Track feature versions and schemas.

    This registry is intentionally lightweight and persists metadata to a JSON
    file so that tests can inspect the resulting state.  A production system
    would likely use a proper metadata database instead.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._registry: dict[str, dict[str, FeatureDescriptor]] = {}
        if path and path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path:
            return
        data = json.loads(self.path.read_text())
        for name, versions in data.items():
            self._registry[name] = {}
            for ver, meta in versions.items():
                self._registry[name][ver] = FeatureDescriptor(
                    name=meta["name"],
                    version=meta["version"],
                    created_at=dt.datetime.fromisoformat(meta["created_at"]),
                    schema=meta["schema"],
                    lineage=meta.get("lineage"),
                )

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if not self.path:
            return
        data: dict[str, dict[str, dict]] = {}
        for name, versions in self._registry.items():
            data[name] = {}
            for ver, desc in versions.items():
                d = asdict(desc)
                d["created_at"] = desc.created_at.isoformat()
                data[name][ver] = d
        self.path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        version: str,
        schema: dict[str, str],
        lineage: dict[str, str] | None = None,
    ) -> FeatureDescriptor:
        desc = FeatureDescriptor(
            name=name,
            version=version,
            created_at=dt.datetime.utcnow(),
            schema=schema,
            lineage=lineage,
        )
        self._registry.setdefault(name, {})[version] = desc
        self._save()
        return desc

    # ------------------------------------------------------------------
    def get(self, name: str, version: str | None = None) -> FeatureDescriptor:
        versions = self._registry.get(name, {})
        if not versions:
            raise KeyError(f"feature '{name}' not registered")
        if version is None:
            # return latest version
            version = sorted(versions.keys())[-1]
        if version not in versions:
            raise KeyError(f"feature '{name}' version '{version}' not found")
        return versions[version]
