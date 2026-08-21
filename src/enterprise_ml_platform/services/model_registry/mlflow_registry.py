"""Model registry backed by the MLflow Model Registry.

This is the registry the serving layer resolves models through. A model
version is an immutable artifact produced by a tracked training run; which
version serves traffic is decided by moving an alias, not by retraining at
load time.

Aliases carry the deployment intent:

``champion``
    The version currently serving production traffic.
``challenger``
    A candidate being evaluated against the champion.

Promotion is therefore a metadata operation -- ``models:/name@champion``
resolves to a different version without the serving layer changing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import structlog

try:  # pragma: no cover - optional dependency
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MlflowException = Exception  # type: ignore

from ...core.exceptions import ServiceError

CHAMPION = "champion"
CHALLENGER = "challenger"

logger = structlog.get_logger(__name__)


class ModelRegistryError(ServiceError):
    """Raised when a registry operation cannot be completed."""


@dataclass(frozen=True)
class ModelVersionInfo:
    """A registered model version.

    Attributes:
        name: Registered model name.
        version: Version number assigned by the registry.
        source: URI of the underlying model artifact.
        run_id: Training run that produced the artifact, if tracked.
        aliases: Deployment aliases currently pointing at this version.
        tags: Free-form metadata attached to the version.
        description: Human readable notes.
        created_at: When the version was registered.
    """

    name: str
    version: str
    source: str
    run_id: str | None = None
    aliases: tuple = ()
    tags: dict[str, str] | None = None
    description: str | None = None
    created_at: dt.datetime | None = None

    @property
    def uri(self) -> str:
        """Return the URI that resolves to exactly this version."""
        return f"models:/{self.name}/{self.version}"


class MLflowModelRegistry:
    """Thin, typed wrapper over the MLflow Model Registry."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """Connect to a registry.

        Args:
            tracking_uri: MLflow tracking URI. Defaults to MLflow's own
                resolution order.
            registry_uri: Model registry URI. Defaults to ``tracking_uri``.

        Raises:
            ModelRegistryError: If MLflow is not installed.
        """
        if mlflow is None:
            raise ModelRegistryError(
                "mlflow is required for the model registry; "
                "install enterprise-ml-platform[training]"
            )
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        resolved_registry = registry_uri or tracking_uri
        if resolved_registry:
            mlflow.set_registry_uri(resolved_registry)
        self.tracking_uri = mlflow.get_tracking_uri()
        self.registry_uri = mlflow.get_registry_uri()
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        self.logger = logger.bind(component="model-registry")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        model_uri: str,
        *,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> ModelVersionInfo:
        """Register ``model_uri`` as a new version of ``name``.

        Args:
            name: Registered model name. Created if it does not exist.
            model_uri: URI of a logged model artifact.
            tags: Metadata to attach to the version.
            description: Human readable notes.

        Raises:
            ModelRegistryError: If registration fails.
        """
        try:
            version = mlflow.register_model(model_uri, name, tags=tags)
            if description:
                self.client.update_model_version(name, version.version, description)
        except MlflowException as exc:
            raise ModelRegistryError(f"could not register {name!r}: {exc}") from exc
        self.logger.info("model_version_registered", name=name, version=version.version)
        return self.get_version(name, str(version.version))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_version(self, name: str, version: str) -> ModelVersionInfo:
        """Return metadata for one version.

        Raises:
            ModelRegistryError: If the version does not exist.
        """
        try:
            mv = self.client.get_model_version(name, version)
        except MlflowException as exc:
            raise ModelRegistryError(
                f"no version {version!r} of model {name!r}: {exc}"
            ) from exc
        return self._to_info(mv)

    def get_by_alias(self, name: str, alias: str = CHAMPION) -> ModelVersionInfo:
        """Return the version an alias currently points at.

        Raises:
            ModelRegistryError: If the alias is not set.
        """
        try:
            mv = self.client.get_model_version_by_alias(name, alias)
        except MlflowException as exc:
            raise ModelRegistryError(
                f"model {name!r} has no {alias!r} alias: {exc}"
            ) from exc
        return self._to_info(mv)

    def list_versions(self, name: str) -> list[ModelVersionInfo]:
        """Return every version of ``name``, newest first."""
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
        except MlflowException as exc:
            raise ModelRegistryError(
                f"could not list versions of {name!r}: {exc}"
            ) from exc
        return sorted(
            (self._to_info(mv) for mv in versions),
            key=lambda info: int(info.version),
            reverse=True,
        )

    def list_models(self) -> list[str]:
        """Return the names of every registered model."""
        return [m.name for m in self.client.search_registered_models()]

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------
    def promote(
        self, name: str, version: str, alias: str = CHAMPION
    ) -> ModelVersionInfo:
        """Point ``alias`` at ``version``.

        This is how a model reaches production: the serving layer resolves
        ``models:/{name}@{alias}``, so moving the alias swaps the served model
        without redeploying anything.
        """
        try:
            self.client.set_registered_model_alias(name, alias, version)
        except MlflowException as exc:
            raise ModelRegistryError(
                f"could not point {alias!r} at {name!r} v{version}: {exc}"
            ) from exc
        self.logger.info("alias_moved", name=name, alias=alias, version=version)
        return self.get_version(name, version)

    def rollback(self, name: str, alias: str = CHAMPION) -> ModelVersionInfo:
        """Move ``alias`` back to the version that preceded the current one.

        Raises:
            ModelRegistryError: If there is nothing to roll back to.
        """
        current = self.get_by_alias(name, alias)
        older = [
            v for v in self.list_versions(name) if int(v.version) < int(current.version)
        ]
        if not older:
            raise ModelRegistryError(
                f"model {name!r} has no version older than v{current.version} to roll back to"
            )
        target = older[0]
        self.logger.warning(
            "alias_rolled_back",
            name=name,
            alias=alias,
            **{"from": current.version, "to": target.version},
        )
        return self.promote(name, target.version, alias)

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias without touching the versions it pointed at."""
        try:
            self.client.delete_registered_model_alias(name, alias)
        except MlflowException as exc:
            raise ModelRegistryError(
                f"could not delete alias {alias!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def resolve_uri(
        self,
        name: str,
        *,
        alias: str | None = None,
        version: str | None = None,
    ) -> str:
        """Return the URI for a version, addressed by alias or by number."""
        if version is not None:
            return f"models:/{name}/{version}"
        return f"models:/{name}@{alias or CHAMPION}"

    def load(
        self,
        name: str,
        *,
        alias: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load a model artifact from the registry.

        Args:
            name: Registered model name.
            alias: Deployment alias to resolve. Defaults to ``champion``.
            version: Exact version, taking precedence over ``alias``.

        Raises:
            ModelRegistryError: If the model cannot be loaded.
        """
        uri = self.resolve_uri(name, alias=alias, version=version)
        try:
            return mlflow.sklearn.load_model(uri)
        except Exception as exc:
            raise ModelRegistryError(f"could not load {uri}: {exc}") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _to_info(mv: Any) -> ModelVersionInfo:
        created = getattr(mv, "creation_timestamp", None)
        return ModelVersionInfo(
            name=mv.name,
            version=str(mv.version),
            source=mv.source,
            run_id=getattr(mv, "run_id", None) or None,
            aliases=tuple(getattr(mv, "aliases", ()) or ()),
            tags=dict(getattr(mv, "tags", {}) or {}),
            description=getattr(mv, "description", None) or None,
            created_at=(
                dt.datetime.fromtimestamp(created / 1000, tz=dt.UTC)
                if created
                else None
            ),
        )
