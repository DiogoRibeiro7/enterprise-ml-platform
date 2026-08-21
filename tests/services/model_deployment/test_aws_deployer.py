"""Tests for the SageMaker deployer.

These run against a stubbed botocore client, so every request is validated
against the real SageMaker API model: a misspelled parameter or a wrong shape
fails here exactly as it would against AWS. What they cannot check is IAM,
quotas, or whether an image actually serves -- that needs an account.
"""

from __future__ import annotations

import boto3
import pytest
from botocore.stub import Stubber

from enterprise_ml_platform.services.model_deployment.deployers import (
    AWSDeployer,
    DeploymentError,
)

REGION = "eu-west-1"
ENDPOINT = "fraud-scoring"
REVISION = "abc123"
ROLE = "arn:aws:iam::123456789012:role/SageMakerExecution"
IMAGE = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn:1.0"
ARTIFACT = "s3://models/fraud/model.tar.gz"
ARN = f"arn:aws:sagemaker:{REGION}:123456789012:endpoint/{ENDPOINT}"
MODEL_ARN = f"arn:aws:sagemaker:{REGION}:123456789012:model/{ENDPOINT}"
CONFIG_ARN = f"arn:aws:sagemaker:{REGION}:123456789012:endpoint-config/{ENDPOINT}"

DEPLOY_CONFIG = {
    "endpoint_name": ENDPOINT,
    "image_uri": IMAGE,
    "revision": REVISION,
    "instance_type": "ml.m5.large",
    "instance_count": 2,
}


@pytest.fixture
def sagemaker():
    """A stubbed SageMaker client that asserts every expected call was made."""
    client = boto3.client(
        "sagemaker",
        region_name=REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    stubber = Stubber(client)
    stubber.activate()
    yield client, stubber
    stubber.assert_no_pending_responses()
    stubber.deactivate()


def _deployer(client, **kwargs) -> AWSDeployer:
    return AWSDeployer(
        REGION, role_arn=ROLE, client=client, poll_interval=0.0, timeout=5.0, **kwargs
    )


def _describe(status: str = "InService", config_name: str = f"{ENDPOINT}-{REVISION}"):
    return {
        "EndpointName": ENDPOINT,
        "EndpointArn": ARN,
        "EndpointConfigName": config_name,
        "EndpointStatus": status,
        "CreationTime": "2024-01-01T00:00:00Z",
        "LastModifiedTime": "2024-01-01T00:00:00Z",
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "CurrentWeight": 1.0,
                "CurrentInstanceCount": 2,
            }
        ],
    }


def _not_found(stubber, operation: str) -> None:
    stubber.add_client_error(
        operation,
        service_error_code="ValidationException",
        service_message=f"Could not find endpoint '{ENDPOINT}'. not found",
        http_status_code=400,
    )


# ----------------------------------------------------------------------
# Creating an endpoint
# ----------------------------------------------------------------------
async def test_deploy_creates_model_config_and_endpoint(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response(
        "create_model",
        {"ModelArn": MODEL_ARN},
        {
            "ModelName": f"{ENDPOINT}-{REVISION}",
            "ExecutionRoleArn": ROLE,
            "PrimaryContainer": {"Image": IMAGE, "ModelDataUrl": ARTIFACT},
        },
    )
    stubber.add_response(
        "create_endpoint_config",
        {"EndpointConfigArn": CONFIG_ARN},
        {
            "EndpointConfigName": f"{ENDPOINT}-{REVISION}",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": f"{ENDPOINT}-{REVISION}",
                    "InitialInstanceCount": 2,
                    "InstanceType": "ml.m5.large",
                    "InitialVariantWeight": 1.0,
                }
            ],
        },
    )
    _not_found(stubber, "describe_endpoint")  # endpoint does not exist yet
    stubber.add_response(
        "create_endpoint",
        {"EndpointArn": ARN},
        {"EndpointName": ENDPOINT, "EndpointConfigName": f"{ENDPOINT}-{REVISION}"},
    )
    stubber.add_response("describe_endpoint", _describe("InService"))

    endpoint = await _deployer(client).deploy(ARTIFACT, DEPLOY_CONFIG)

    assert endpoint == ENDPOINT


async def test_deploy_updates_an_existing_endpoint_in_place(sagemaker) -> None:
    """Redeploying must not tear the endpoint down and lose its traffic."""
    client, stubber = sagemaker
    stubber.add_response("create_model", {"ModelArn": MODEL_ARN})
    stubber.add_response("create_endpoint_config", {"EndpointConfigArn": CONFIG_ARN})
    stubber.add_response("describe_endpoint", _describe(config_name="old-config"))
    stubber.add_response(
        "update_endpoint",
        {"EndpointArn": ARN},
        {"EndpointName": ENDPOINT, "EndpointConfigName": f"{ENDPOINT}-{REVISION}"},
    )
    stubber.add_response("describe_endpoint", _describe("InService"))

    endpoint = await _deployer(client).deploy(ARTIFACT, DEPLOY_CONFIG)

    assert endpoint == ENDPOINT


async def test_deploy_waits_for_the_endpoint_to_serve(sagemaker) -> None:
    """Create returns immediately; the endpoint is not usable until InService."""
    client, stubber = sagemaker
    stubber.add_response("create_model", {"ModelArn": MODEL_ARN})
    stubber.add_response("create_endpoint_config", {"EndpointConfigArn": CONFIG_ARN})
    _not_found(stubber, "describe_endpoint")
    stubber.add_response("create_endpoint", {"EndpointArn": ARN})
    stubber.add_response("describe_endpoint", _describe("Creating"))
    stubber.add_response("describe_endpoint", _describe("Creating"))
    stubber.add_response("describe_endpoint", _describe("InService"))

    assert await _deployer(client).deploy(ARTIFACT, DEPLOY_CONFIG) == ENDPOINT


async def test_deploy_fails_when_the_endpoint_fails(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("create_model", {"ModelArn": MODEL_ARN})
    stubber.add_response("create_endpoint_config", {"EndpointConfigArn": CONFIG_ARN})
    _not_found(stubber, "describe_endpoint")
    stubber.add_response("create_endpoint", {"EndpointArn": ARN})
    stubber.add_response("describe_endpoint", _describe("Failed"))

    with pytest.raises(DeploymentError, match="Failed"):
        await _deployer(client).deploy(ARTIFACT, DEPLOY_CONFIG)


async def test_deploy_times_out_rather_than_hanging(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("create_model", {"ModelArn": MODEL_ARN})
    stubber.add_response("create_endpoint_config", {"EndpointConfigArn": CONFIG_ARN})
    _not_found(stubber, "describe_endpoint")
    stubber.add_response("create_endpoint", {"EndpointArn": ARN})
    stubber.add_response("describe_endpoint", _describe("Creating"))

    deployer = AWSDeployer(
        REGION, role_arn=ROLE, client=client, poll_interval=0.0, timeout=-1.0
    )
    with pytest.raises(DeploymentError, match="still Creating"):
        await deployer.deploy(ARTIFACT, DEPLOY_CONFIG)


# ----------------------------------------------------------------------
# Required configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing,message",
    [
        ("endpoint_name", "endpoint_name"),
        ("image_uri", "image_uri"),
    ],
)
async def test_missing_configuration_is_refused_before_calling_aws(
    sagemaker, missing, message
) -> None:
    client, _ = sagemaker
    config = {k: v for k, v in DEPLOY_CONFIG.items() if k != missing}

    with pytest.raises(DeploymentError, match=message):
        await _deployer(client).deploy(ARTIFACT, config)


async def test_missing_execution_role_is_refused(sagemaker) -> None:
    client, _ = sagemaker
    deployer = AWSDeployer(REGION, client=client, poll_interval=0.0)

    with pytest.raises(DeploymentError, match="execution role"):
        await deployer.deploy(ARTIFACT, DEPLOY_CONFIG)


# ----------------------------------------------------------------------
# Traffic
# ----------------------------------------------------------------------
async def test_update_traffic_sets_variant_weights(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response(
        "update_endpoint_weights_and_capacities",
        {"EndpointArn": ARN},
        {
            "EndpointName": ENDPOINT,
            "DesiredWeightsAndCapacities": [
                {"VariantName": "blue", "DesiredWeight": 0.9},
                {"VariantName": "green", "DesiredWeight": 0.1},
            ],
        },
    )
    stubber.add_response("describe_endpoint", _describe("InService"))

    await _deployer(client).update_traffic(ENDPOINT, {"blue": 0.9, "green": 0.1})


async def test_update_traffic_without_variants_is_refused(sagemaker) -> None:
    client, _ = sagemaker

    with pytest.raises(DeploymentError, match="at least one variant"):
        await _deployer(client).update_traffic(ENDPOINT, {})


# ----------------------------------------------------------------------
# Rollback
# ----------------------------------------------------------------------
async def test_rollback_points_the_endpoint_at_a_named_config(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response(
        "update_endpoint",
        {"EndpointArn": ARN},
        {"EndpointName": ENDPOINT, "EndpointConfigName": "previous-config"},
    )
    stubber.add_response("describe_endpoint", _describe("InService"))

    await _deployer(client).rollback(ENDPOINT, "previous-config")


async def test_rollback_falls_back_to_the_last_other_config(sagemaker) -> None:
    """Endpoint configs are immutable, so the previous one is still there."""
    client, stubber = sagemaker
    stubber.add_response("describe_endpoint", _describe(config_name="current"))
    stubber.add_response(
        "list_endpoint_configs",
        {
            "EndpointConfigs": [
                {
                    "EndpointConfigName": "current",
                    "EndpointConfigArn": CONFIG_ARN,
                    "CreationTime": "2024-02-01T00:00:00Z",
                },
                {
                    "EndpointConfigName": "previous",
                    "EndpointConfigArn": CONFIG_ARN,
                    "CreationTime": "2024-01-01T00:00:00Z",
                },
            ]
        },
    )
    stubber.add_response(
        "update_endpoint",
        {"EndpointArn": ARN},
        {"EndpointName": ENDPOINT, "EndpointConfigName": "previous"},
    )
    stubber.add_response("describe_endpoint", _describe("InService"))

    await _deployer(client).rollback(ENDPOINT)


async def test_rollback_without_an_earlier_config_is_refused(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("describe_endpoint", _describe(config_name="only"))
    stubber.add_response(
        "list_endpoint_configs",
        {
            "EndpointConfigs": [
                {
                    "EndpointConfigName": "only",
                    "EndpointConfigArn": CONFIG_ARN,
                    "CreationTime": "2024-01-01T00:00:00Z",
                }
            ]
        },
    )

    with pytest.raises(DeploymentError, match="no earlier configuration"):
        await _deployer(client).rollback(ENDPOINT)


# ----------------------------------------------------------------------
# Health and deletion
# ----------------------------------------------------------------------
async def test_check_health_reads_the_control_plane(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("describe_endpoint", _describe("InService"))

    assert await _deployer(client).check_health(ENDPOINT) is True


async def test_check_health_is_false_while_updating(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("describe_endpoint", _describe("Updating"))

    assert await _deployer(client).check_health(ENDPOINT) is False


async def test_check_health_is_false_for_a_missing_endpoint(sagemaker) -> None:
    client, stubber = sagemaker
    _not_found(stubber, "describe_endpoint")

    assert await _deployer(client).check_health(ENDPOINT) is False


async def test_delete_removes_the_endpoint(sagemaker) -> None:
    client, stubber = sagemaker
    stubber.add_response("delete_endpoint", {}, {"EndpointName": ENDPOINT})

    await _deployer(client).delete(ENDPOINT)


async def test_deleting_a_missing_endpoint_is_not_an_error(sagemaker) -> None:
    client, stubber = sagemaker
    _not_found(stubber, "delete_endpoint")

    await _deployer(client).delete(ENDPOINT)


# ----------------------------------------------------------------------
# Threading
# ----------------------------------------------------------------------
async def test_blocking_aws_calls_run_off_the_event_loop(sagemaker) -> None:
    """boto3 is synchronous; calling it inline would stall every request."""
    import threading

    client, stubber = sagemaker
    observed: dict = {}
    original = client.describe_endpoint

    def recording(**kwargs):
        observed["thread"] = threading.current_thread().name
        return original(**kwargs)

    client.describe_endpoint = recording
    stubber.add_response("describe_endpoint", _describe("InService"))

    await _deployer(client).describe(ENDPOINT)

    assert observed["thread"] != threading.current_thread().name
