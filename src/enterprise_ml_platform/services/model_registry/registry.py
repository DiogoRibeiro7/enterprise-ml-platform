from __future__ import annotations

"""High level model registry service orchestrating sub-components."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from .versioning.version_manager import VersionManager
from .metadata.metadata_store import MetadataStore
from .lineage.lineage_tracker import LineageTracker
from .governance.model_governance import ModelGovernance
from .storage.artifact_store import ArtifactStore
from .search.model_search import ModelSearch
from .comparison.model_comparator import ModelComparator
from .export.model_exporter import ModelExporter


@dataclass
class ModelRegistry:
    """Central point for model version management and discovery."""

    version_manager: VersionManager = field(default_factory=VersionManager)
    metadata_store: MetadataStore = field(default_factory=MetadataStore)
    lineage_tracker: LineageTracker = field(default_factory=LineageTracker)
    governance: ModelGovernance = field(default_factory=ModelGovernance)
    artifact_store: ArtifactStore = field(default_factory=ArtifactStore)
    search_engine: ModelSearch = field(default_factory=ModelSearch)
    comparator: ModelComparator = field(default_factory=ModelComparator)
    exporter: ModelExporter = field(default_factory=ModelExporter)

    metrics: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    def register(
        self,
        name: str,
        model: Any,
        metadata: Dict[str, Any] | None = None,
        metrics: Dict[str, float] | None = None,
        parents: List[Tuple[str, str]] | None = None,
        datasets: List[str] | None = None,
    ) -> str:
        """Register a new model version and persist related information."""

        version = self.version_manager.next_version(name)
        self.metadata_store.save(name, version, metadata or {})
        if metrics:
            self.metrics[(name, version)] = metrics
        self.lineage_tracker.record(name, version, parents, datasets)
        self.artifact_store.save(name, version, model)
        self.governance.set_stage(name, version, "development")
        if mlflow:
            mlflow.log_params({"model": name, "version": version})
        return version

    def get(self, name: str, version: str | None = None) -> Any:
        """Return stored model artifact for ``name`` and ``version``."""

        if version is None:
            version = self.version_manager.versions.get(name, [None])[-1]
        return self.artifact_store.get(name, version) if version else None

    def promote(self, name: str, version: str, stage: str) -> None:
        """Move a model version to a new governance stage."""

        self.governance.set_stage(name, version, stage)

    def search(self, query: str) -> List[Tuple[str, str]]:
        """Search across metadata."""

        return self.search_engine.search(self.metadata_store.store, query)

    def compare(
        self,
        model_a: Tuple[str, str],
        model_b: Tuple[str, str],
    ) -> Dict[str, float]:
        """Compare two model versions based on logged metrics."""

        return self.comparator.compare(self.metrics, model_a, model_b)

    def export(self, name: str, version: str, fmt: str) -> str:
        """Export a model artifact to a different format."""

        model = self.artifact_store.get(name, version)
        return self.exporter.export(model, fmt)

    def auto_promote(self, name: str, version: str, metric: str, threshold: float) -> None:
        """Automatically promote model if metric exceeds ``threshold``."""

        score = self.metrics.get((name, version), {}).get(metric, 0.0)
        if score >= threshold:
            self.promote(name, version, "production")
