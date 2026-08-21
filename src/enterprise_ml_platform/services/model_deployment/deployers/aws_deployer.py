"""AWS SageMaker deployer.

Deploying to SageMaker is three resources, not one: a *model* (an artifact
plus the container that serves it), an *endpoint configuration* (which models
sit behind the endpoint, on what hardware, at what traffic weights), and the
*endpoint* itself. Only the last one is addressable; the first two are
immutable, which is what makes rollback possible -- the previous endpoint
configuration is still there, so reverting means pointing the endpoint back at
it rather than rebuilding anything.

Every call here is a blocking boto3 call dispatched to a worker thread, and
endpoint transitions are polled to completion, because SageMaker's create and
update operations return long before the endpoint can serve traffic.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import structlog

try:  # pragma: no cover - optional dependency
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover
    boto3 = None

    class ClientError(Exception):  # type: ignore
        """Stand-in so the module imports without botocore."""


#: Endpoint states that mean SageMaker is still working.
IN_FLIGHT_STATUSES = frozenset(
    {"Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"}
)
READY_STATUS = "InService"
FAILED_STATUSES = frozenset({"Failed", "OutOfService"})


class DeploymentError(RuntimeError):
    """Raised when a SageMaker deployment cannot be completed."""


@dataclass(frozen=True)
class EndpointState:
    """A snapshot of a SageMaker endpoint.

    Attributes:
        name: Endpoint name, the identifier every SageMaker API takes.
        arn: Full ARN of the endpoint.
        status: SageMaker endpoint status.
        config_name: Endpoint configuration currently in effect.
        variants: Variant name to current traffic weight.
    """

    name: str
    arn: str
    status: str
    config_name: str
    variants: dict[str, float]

    @property
    def is_ready(self) -> bool:
        """Return True when the endpoint can serve traffic."""
        return self.status == READY_STATUS


class AWSDeployer:
    """Deploy models to AWS SageMaker real-time endpoints."""

    def __init__(
        self,
        region: str,
        role_arn: str | None = None,
        *,
        client: Any = None,
        poll_interval: float = 15.0,
        timeout: float = 1800.0,
    ) -> None:
        """Create a deployer.

        Args:
            region: AWS region the endpoint lives in.
            role_arn: Execution role SageMaker assumes to read the artifact
                and write logs. Required to create models.
            client: Pre-built SageMaker client. Built from ``region`` if absent.
            poll_interval: Seconds between endpoint status checks.
            timeout: Seconds to wait for an endpoint to reach ``InService``.

        Raises:
            RuntimeError: If boto3 is unavailable and no client is supplied.
        """
        if client is None and boto3 is None:  # pragma: no cover - env lacks boto3
            raise RuntimeError(
                "boto3 is required for AWS deployments; "
                "install enterprise-ml-platform[aws]"
            )
        self.region = region
        self.role_arn = role_arn
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._client = client
        self.logger = structlog.get_logger().bind(platform="aws", region=region)

    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Return the SageMaker client, building it on first use."""
        if self._client is None:  # pragma: no cover - requires boto3 + credentials
            self._client = boto3.client("sagemaker", region_name=self.region)
        return self._client

    async def _call(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """Run one blocking SageMaker call on a worker thread."""
        method = getattr(self.client, operation)
        response: dict[str, Any] = await asyncio.to_thread(lambda: method(**kwargs))
        return response

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------
    async def deploy(self, model_path: str, config: dict[str, Any]) -> str:
        """Deploy ``model_path`` to a SageMaker endpoint.

        Creates the model and a new endpoint configuration, then either
        creates the endpoint or updates it in place if it already exists, and
        waits until it is serving.

        Args:
            model_path: S3 URI of the model artifact (``s3://.../model.tar.gz``).
            config: Deployment options. ``endpoint_name`` and ``image_uri`` are
                required; ``instance_type``, ``instance_count``,
                ``variant_name``, ``revision`` and ``role_arn`` are optional.

        Returns:
            The endpoint name. SageMaker endpoints are invoked through the
            signed runtime API, so there is no plain URL to hand back.

        Raises:
            DeploymentError: If required options are missing or SageMaker
                rejects the deployment.
        """
        endpoint_name: str | None = config.get("endpoint_name")
        image_uri = config.get("image_uri")
        role_arn = config.get("role_arn") or self.role_arn
        if not endpoint_name:
            raise DeploymentError("config['endpoint_name'] is required")
        if not image_uri:
            raise DeploymentError("config['image_uri'] is required")
        if not role_arn:
            raise DeploymentError("an execution role ARN is required to create a model")

        revision = str(config.get("revision") or uuid.uuid4().hex[:12])
        model_name = f"{endpoint_name}-{revision}"
        config_name = f"{endpoint_name}-{revision}"
        variant_name = config.get("variant_name", "AllTraffic")

        log = self.logger.bind(endpoint=endpoint_name, revision=revision)
        log.info("creating-model", model_path=model_path)
        await self._call(
            "create_model",
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_path},
        )

        log.info("creating-endpoint-config")
        await self._call(
            "create_endpoint_config",
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": variant_name,
                    "ModelName": model_name,
                    "InitialInstanceCount": int(config.get("instance_count", 1)),
                    "InstanceType": config.get("instance_type", "ml.m5.large"),
                    "InitialVariantWeight": 1.0,
                }
            ],
        )

        if await self._endpoint_exists(endpoint_name):
            log.info("updating-endpoint")
            await self._call(
                "update_endpoint",
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            log.info("creating-endpoint")
            await self._call(
                "create_endpoint",
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )

        state = await self.wait_until_ready(endpoint_name)
        log.info("endpoint-ready", arn=state.arn)
        return endpoint_name

    # ------------------------------------------------------------------
    async def _endpoint_exists(self, endpoint: str) -> bool:
        """Return whether the endpoint is already present."""
        try:
            await self._call("describe_endpoint", EndpointName=endpoint)
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    # ------------------------------------------------------------------
    async def describe(self, endpoint: str) -> EndpointState:
        """Return the current state of ``endpoint``.

        Raises:
            DeploymentError: If the endpoint does not exist.
        """
        try:
            response = await self._call("describe_endpoint", EndpointName=endpoint)
        except ClientError as exc:
            if _is_not_found(exc):
                raise DeploymentError(f"endpoint {endpoint!r} does not exist") from exc
            raise
        return EndpointState(
            name=response["EndpointName"],
            arn=response.get("EndpointArn", ""),
            status=response["EndpointStatus"],
            config_name=response.get("EndpointConfigName", ""),
            variants={
                variant["VariantName"]: float(variant.get("CurrentWeight", 0.0))
                for variant in response.get("ProductionVariants", [])
            },
        )

    # ------------------------------------------------------------------
    async def wait_until_ready(self, endpoint: str) -> EndpointState:
        """Poll until ``endpoint`` is serving traffic.

        SageMaker's create and update calls return as soon as the request is
        accepted, long before the endpoint can answer.

        Raises:
            DeploymentError: If the endpoint fails or the timeout elapses.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            state = await self.describe(endpoint)
            if state.is_ready:
                return state
            if state.status in FAILED_STATUSES:
                raise DeploymentError(
                    f"endpoint {endpoint!r} entered status {state.status}"
                )
            if time.monotonic() >= deadline:
                raise DeploymentError(
                    f"endpoint {endpoint!r} still {state.status} after "
                    f"{self.timeout:.0f}s"
                )
            await asyncio.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    async def check_health(self, endpoint: str) -> bool:
        """Return whether ``endpoint`` is currently serving.

        SageMaker endpoints are not plain HTTP URLs, so readiness is read from
        the control plane rather than fetched.
        """
        try:
            return (await self.describe(endpoint)).is_ready
        except DeploymentError:
            return False

    # ------------------------------------------------------------------
    # Traffic and rollback
    # ------------------------------------------------------------------
    async def update_traffic(self, endpoint: str, variants: dict[str, float]) -> None:
        """Shift traffic across the endpoint's production variants.

        Args:
            endpoint: Endpoint name.
            variants: Variant name to desired weight. Weights are relative,
                so ``{"blue": 0.9, "green": 0.1}`` sends a tenth to green.

        Raises:
            DeploymentError: If no variants are supplied.
        """
        if not variants:
            raise DeploymentError("at least one variant weight is required")
        self.logger.info("update-traffic", endpoint=endpoint, variants=variants)
        await self._call(
            "update_endpoint_weights_and_capacities",
            EndpointName=endpoint,
            DesiredWeightsAndCapacities=[
                {"VariantName": name, "DesiredWeight": float(weight)}
                for name, weight in variants.items()
            ],
        )
        await self.wait_until_ready(endpoint)

    # ------------------------------------------------------------------
    async def list_configs(self, endpoint: str) -> list[str]:
        """Return this endpoint's configurations, newest first."""
        response = await self._call(
            "list_endpoint_configs",
            NameContains=endpoint,
            SortBy="CreationTime",
            SortOrder="Descending",
        )
        return [c["EndpointConfigName"] for c in response.get("EndpointConfigs", [])]

    async def rollback(
        self, endpoint: str, previous_version: str | None = None
    ) -> None:
        """Point ``endpoint`` back at an earlier endpoint configuration.

        Args:
            endpoint: Endpoint name.
            previous_version: Endpoint configuration to revert to. Defaults to
                the most recent one that is not currently in effect.

        Raises:
            DeploymentError: If there is no earlier configuration to revert to.
        """
        target = previous_version
        if target is None:
            current = (await self.describe(endpoint)).config_name
            candidates = [c for c in await self.list_configs(endpoint) if c != current]
            if not candidates:
                raise DeploymentError(
                    f"endpoint {endpoint!r} has no earlier configuration to roll back to"
                )
            target = candidates[0]

        self.logger.warning("rollback", endpoint=endpoint, config=target)
        await self._call(
            "update_endpoint", EndpointName=endpoint, EndpointConfigName=target
        )
        await self.wait_until_ready(endpoint)

    # ------------------------------------------------------------------
    async def delete(self, endpoint: str) -> None:
        """Delete ``endpoint``. Its configurations and models are left intact.

        Keeping them is deliberate: they are the rollback targets, and they
        cost nothing while no endpoint references them.
        """
        self.logger.info("delete-endpoint", endpoint=endpoint)
        try:
            await self._call("delete_endpoint", EndpointName=endpoint)
        except ClientError as exc:
            if not _is_not_found(exc):
                raise


def _is_not_found(exc: ClientError) -> bool:
    """Return whether a ClientError means the resource does not exist."""
    response = getattr(exc, "response", {}) or {}
    error = response.get("Error", {})
    if error.get("Code") in {"ValidationException", "ResourceNotFound"}:
        return (
            "not found" in error.get("Message", "").lower()
            or error.get("Code") == "ResourceNotFound"
        )
    return False
