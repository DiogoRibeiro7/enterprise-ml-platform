"""One dataset carried through the whole lifecycle.

Every other example in this repository shows a single component. This one runs
the sequence the platform exists for, end to end, on one synthetic dataset:

    features -> offline store -> point-in-time training set -> training
    -> tracked run -> registry -> promotion -> serving -> rollback

Two properties are what separate this from a training script, and both are
visible in the output:

**Point-in-time retrieval.** Labels are observed at particular moments. The
training set is assembled by asking the offline store what each customer
looked like *at that moment*, not what they look like now. The run prints a
row where the two differ, which is the leak the discipline prevents.

**Promotion is a metadata operation.** The serving layer resolves
``models:/fraud-scorer@champion``. Moving the alias changes which model
answers, with no redeployment and no code change, and rollback moves it back.
The version in each prediction response is the evidence.

Nothing external is required: the offline store is Parquet on disk, MLflow
tracks to SQLite, and the API runs in-process. Everything is written to a
temporary directory unless ``--workdir`` says otherwise.

Run it with::

    python -m examples.fraud_detection.end_to_end
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
import tempfile

# MLflow narrates its own work at INFO, draws an artifact progress bar per
# logged model, and reads the progress-bar setting when it is imported. Both
# have to be silenced before that import, which is why everything below this
# block is a deliberately late import.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.tracking._model_registry.client").setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import structlog  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from enterprise_ml_platform.api.config import APISettings  # noqa: E402
from enterprise_ml_platform.api.main import create_app  # noqa: E402
from enterprise_ml_platform.services.feature_store import (  # noqa: E402
    ParquetOfflineStore,
)
from enterprise_ml_platform.services.model_registry import (  # noqa: E402
    CHAMPION,
    MLflowModelRegistry,
)
from enterprise_ml_platform.services.model_training import (  # noqa: E402
    ModelConfig,
    ModelTrainingService,
)

FEATURE_SET = "customer_risk"
FEATURE_SET_VERSION = "v1"
MODEL_NAME = "fraud-scorer"
FEATURES = ("avg_amount_30d", "txn_count_30d", "days_since_signup")
API_KEY = "example-key"

#: Monthly snapshots. Behaviour drifts between them, which is what makes an
#: as-of lookup differ from the latest row.
SNAPSHOT_DATES = pd.date_range("2024-01-01", periods=6, freq="MS")
SEED = 20240601


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
def build_snapshots(rng: np.random.Generator, customers: int = 200) -> pd.DataFrame:
    """Return a feature snapshot per customer per month."""
    rows = []
    for customer in range(customers):
        spend = rng.lognormal(4.0, 0.6)
        rate = rng.uniform(2, 25)
        tenure = rng.integers(30, 900)
        for step, when in enumerate(SNAPSHOT_DATES):
            rows.append(
                {
                    "entity_id": f"C{customer:04d}",
                    "timestamp": when,
                    "avg_amount_30d": round(
                        spend * (1 + 0.35 * step) + rng.normal(0, 3), 2
                    ),
                    "txn_count_30d": float(
                        max(0, round(rate + 4 * step + rng.normal(0, 2)))
                    ),
                    "days_since_signup": float(tenure + 30 * step),
                }
            )
    return pd.DataFrame(rows)


def build_labels(
    rng: np.random.Generator, snapshots: pd.DataFrame, count: int = 600
) -> pd.DataFrame:
    """Return fraud outcomes, each observed at its own moment.

    Decisions fall between the second and fifth snapshot, so for almost every
    row there is a later snapshot that a naive join would wrongly pull in.
    """
    identifiers = snapshots["entity_id"].unique()
    start, end = SNAPSHOT_DATES[1], SNAPSHOT_DATES[4]
    span = (end - start).days

    rows = []
    for _ in range(count):
        entity = str(rng.choice(identifiers))
        when = start + pd.Timedelta(days=int(rng.integers(0, span)))
        history = snapshots[
            (snapshots["entity_id"] == entity) & (snapshots["timestamp"] <= when)
        ]
        if history.empty:
            continue
        state = history.sort_values("timestamp").iloc[-1]
        # A logistic on the drivers, tuned to a plausible base rate.
        odds = (
            -3.1
            + 0.9 * (state["txn_count_30d"] - 20) / 12
            + 0.8 * (state["avg_amount_30d"] - 90) / 60
            - 0.5 * (state["days_since_signup"] - 500) / 300
        )
        rows.append(
            {
                "entity_id": entity,
                "decision_time": when,
                "is_fraud": int(rng.uniform() < 1 / (1 + np.exp(-odds))),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Point-in-time assembly
# ---------------------------------------------------------------------------
async def assemble_training_set(
    store: ParquetOfflineStore, labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, int]:
    """Join each label to the features as they stood when it was observed.

    Returns the design matrix, the targets, and how many rows would have
    carried a different value had the latest snapshot been used instead.
    """
    rows: list[list[float]] = []
    targets: list[int] = []
    would_have_leaked = 0

    for _, label in labels.iterrows():
        as_of = await store.get_features(
            FEATURE_SET,
            FEATURE_SET_VERSION,
            label["entity_id"],
            as_of=label["decision_time"],
        )
        if not all(name in as_of for name in FEATURES):
            continue
        latest = await store.get_features(
            FEATURE_SET, FEATURE_SET_VERSION, label["entity_id"]
        )
        if any(as_of[name] != latest.get(name) for name in FEATURES):
            would_have_leaked += 1
        rows.append([as_of[name] for name in FEATURES])
        targets.append(int(label["is_fraud"]))

    return np.array(rows, dtype=float), np.array(targets, dtype=int), would_have_leaked


async def show_one_leak(store: ParquetOfflineStore, labels: pd.DataFrame) -> None:
    """Print a single row where as-of and latest disagree."""
    for _, label in labels.iterrows():
        as_of = await store.get_features(
            FEATURE_SET,
            FEATURE_SET_VERSION,
            label["entity_id"],
            as_of=label["decision_time"],
        )
        latest = await store.get_features(
            FEATURE_SET, FEATURE_SET_VERSION, label["entity_id"]
        )
        if as_of.get("txn_count_30d") != latest.get("txn_count_30d"):
            print(
                f"    customer {label['entity_id']}, decision {label['decision_time'].date()}"
            )
            print(f"      txn_count_30d as of that day : {as_of['txn_count_30d']:.0f}")
            print(
                f"      txn_count_30d today          : {latest['txn_count_30d']:.0f}  <- the future"
            )
            return


# ---------------------------------------------------------------------------
# Training and the registry
# ---------------------------------------------------------------------------
async def train_candidate(
    tracking_uri: str,
    artifact_location: str,
    features: np.ndarray,
    targets: np.ndarray,
    estimators: list,
) -> tuple[str | None, dict[str, float]]:
    """Train one candidate inside a tracked MLflow run."""
    service = ModelTrainingService(
        tracking_uri=tracking_uri,
        experiment_name="fraud-detection",
        artifact_location=artifact_location,
    )
    config = ModelConfig(
        algorithm="ensemble",
        ensemble={
            "estimators": estimators,
            "task": "classification",
            # Soft voting exposes scores, without which there is no ROC AUC.
            "method": "voting",
            "params": {"voting": "soft"},
        },
    )
    _, metrics = await service.train(features, targets, config)
    return service.last_model_uri, metrics


def report(label: str, metrics: dict[str, float]) -> None:
    baseline = metrics.get("majority_class_rate", float("nan"))
    print(
        f"    {label:<12} accuracy {metrics['accuracy']:.3f} "
        f"(always-no scores {baseline:.3f})"
        f"  recall {metrics['recall']:.3f}"
        f"  roc_auc {metrics.get('roc_auc', float('nan')):.3f}"
    )


def predict_once(client: TestClient, row: list[float]) -> dict:
    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": API_KEY},
        json={"model_name": MODEL_NAME, "features": row},
    )
    response.raise_for_status()
    return response.json()


def reload_and_predict(client: TestClient, row: list[float]) -> dict:
    """Pull whatever the champion alias now points at, then score a row."""
    client.post(f"/api/v1/models/{MODEL_NAME}/load", headers={"X-API-Key": API_KEY})
    return predict_once(client, row)


# ---------------------------------------------------------------------------
async def run(workdir: pathlib.Path) -> None:
    """Carry one dataset through the platform, narrating each step."""
    # MLflow writes some notices straight to unbuffered stderr. Without this,
    # a piped stdout is block-buffered and its lines arrive out of order.
    sys.stdout.reconfigure(line_buffering=True)

    # The services log their own progress; this example narrates instead.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    tracking_uri = f"sqlite:///{(workdir / 'mlflow.db').as_posix()}"
    artifact_location = (workdir / "artifacts").as_uri()
    rng = np.random.default_rng(SEED)

    print(f"\nworking directory: {workdir}\n")

    print("1. generate a dataset")
    snapshots = build_snapshots(rng)
    labels = build_labels(rng, snapshots)
    print(f"    {len(snapshots)} feature snapshots over {len(SNAPSHOT_DATES)} months")
    print(
        f"    {len(labels)} labelled decisions, {labels['is_fraud'].mean():.1%} fraud\n"
    )

    print("2. write the feature set to the offline store")
    offline = ParquetOfflineStore(workdir / "offline")
    await offline.write_features(FEATURE_SET, FEATURE_SET_VERSION, snapshots)
    path = offline.path_for(FEATURE_SET, FEATURE_SET_VERSION)
    print(f"    {path.relative_to(workdir)}\n")

    print("3. assemble the training set as of each decision")
    features, targets, leaky_rows = await assemble_training_set(offline, labels)
    print(f"    {features.shape[0]} rows x {features.shape[1]} features")
    print(f"    {leaky_rows} of them would carry a future value under a naive join:")
    await show_one_leak(offline, labels)
    print()

    print("4. train two candidates, each in a tracked run")
    baseline_uri, baseline_metrics = await train_candidate(
        tracking_uri,
        artifact_location,
        features,
        targets,
        [("lr", LogisticRegression(max_iter=400))],
    )
    report("baseline", baseline_metrics)
    challenger_uri, challenger_metrics = await train_candidate(
        tracking_uri,
        artifact_location,
        features,
        targets,
        [
            ("lr", LogisticRegression(max_iter=400)),
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ],
    )
    report("challenger", challenger_metrics)
    print("    accuracy barely separates them; recall and roc_auc do")
    print()

    print("5. register both, promote the baseline")
    registry = MLflowModelRegistry(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    baseline = registry.register(MODEL_NAME, baseline_uri)
    challenger = registry.register(MODEL_NAME, challenger_uri)
    registry.promote(MODEL_NAME, baseline.version, CHAMPION)
    print(f"    versions {baseline.version} and {challenger.version} registered")
    print(f"    champion -> v{baseline.version}\n")

    print("6. serve it")
    app = create_app(
        APISettings(
            environment="production",
            api_key=API_KEY,
            allow_demo_models=False,
            model_registry_uri=tracking_uri,
        )
    )
    client = TestClient(app)
    row = features[0].tolist()
    served = reload_and_predict(client, row)
    print(
        f"    POST /api/v1/predict -> version {served['model_version']}, "
        f"{served['latency_ms']:.1f} ms\n"
    )

    print("7. promote the challenger, without redeploying anything")
    registry.promote(MODEL_NAME, challenger.version, CHAMPION)
    served = reload_and_predict(client, row)
    print(f"    same endpoint, same request -> version {served['model_version']}\n")

    print("8. roll back")
    registry.rollback(MODEL_NAME, CHAMPION)
    served = reload_and_predict(client, row)
    print(f"    same endpoint, same request -> version {served['model_version']}\n")

    print("both versions still exist; only the alias moved:")
    for version in registry.list_versions(MODEL_NAME):
        marker = " <- champion" if CHAMPION in version.aliases else ""
        print(f"    v{version.version}  run {version.run_id[:8]}{marker}")
    print()


def main() -> None:
    """Parse arguments and run the walkthrough."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir",
        type=pathlib.Path,
        default=None,
        help="where to write the stores (default: a temporary directory)",
    )
    args = parser.parse_args()

    workdir = args.workdir or pathlib.Path(tempfile.mkdtemp(prefix="fraud-e2e-"))
    workdir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(workdir))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
