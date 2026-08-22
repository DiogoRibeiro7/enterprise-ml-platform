"""The end-to-end example has to keep running.

An example that stops working is worse than none: it is the first thing a
reader tries, and the README points at it as the demonstration that the
components work together. This runs the whole walkthrough and checks the two
properties it exists to show.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from examples.fraud_detection import end_to_end as example


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(example.SEED)
    snapshots = example.build_snapshots(rng, customers=40)
    labels = example.build_labels(rng, snapshots, count=120)
    return snapshots, labels


# ----------------------------------------------------------------------
# The dataset is worth learning from
# ----------------------------------------------------------------------
def test_snapshots_cover_every_customer_and_month(dataset) -> None:
    snapshots, _ = dataset

    assert len(snapshots) == 40 * len(example.SNAPSHOT_DATES)
    assert set(snapshots.columns) == {"entity_id", "timestamp", *example.FEATURES}


def test_the_fraud_rate_is_plausible(dataset) -> None:
    """A near-balanced or near-empty target would teach the wrong lesson."""
    _, labels = dataset

    assert 0.02 < labels["is_fraud"].mean() < 0.30


def test_decisions_fall_before_the_last_snapshot(dataset) -> None:
    """Otherwise as-of and latest coincide and the example proves nothing."""
    _, labels = dataset

    assert labels["decision_time"].max() < example.SNAPSHOT_DATES[-1]


def test_the_dataset_is_deterministic() -> None:
    first = example.build_snapshots(np.random.default_rng(example.SEED), customers=5)
    second = example.build_snapshots(np.random.default_rng(example.SEED), customers=5)

    assert first.equals(second)


# ----------------------------------------------------------------------
# Point-in-time assembly
# ----------------------------------------------------------------------
async def test_training_set_is_built_from_point_in_time_features(
    tmp_path: pathlib.Path, dataset
) -> None:
    snapshots, labels = dataset
    store = example.ParquetOfflineStore(tmp_path / "offline")
    await store.write_features(
        example.FEATURE_SET, example.FEATURE_SET_VERSION, snapshots
    )

    features, targets, would_have_leaked = await example.assemble_training_set(
        store, labels
    )

    assert features.shape == (len(targets), len(example.FEATURES))
    assert set(np.unique(targets)) <= {0, 1}
    # The whole point: a naive join would have pulled a later value in.
    assert would_have_leaked > 0.8 * len(targets)


async def test_no_assembled_row_carries_a_value_from_after_its_decision(
    tmp_path: pathlib.Path, dataset
) -> None:
    """Checked directly against the source data, not through the store."""
    snapshots, labels = dataset
    store = example.ParquetOfflineStore(tmp_path / "offline")
    await store.write_features(
        example.FEATURE_SET, example.FEATURE_SET_VERSION, snapshots
    )

    for _, label in labels.head(25).iterrows():
        served = await store.get_features(
            example.FEATURE_SET,
            example.FEATURE_SET_VERSION,
            label["entity_id"],
            as_of=label["decision_time"],
        )
        eligible = snapshots[
            (snapshots["entity_id"] == label["entity_id"])
            & (snapshots["timestamp"] <= label["decision_time"])
        ].sort_values("timestamp")
        expected = eligible.iloc[-1]

        for name in example.FEATURES:
            assert served[name] == pytest.approx(expected[name])


# ----------------------------------------------------------------------
# The whole walkthrough
# ----------------------------------------------------------------------
@pytest.mark.slow
async def test_the_walkthrough_runs(tmp_path: pathlib.Path, capsys) -> None:
    """Run it exactly as a reader would, and read the narrative back."""
    await example.run(tmp_path)
    out = capsys.readouterr().out

    # Each numbered step reported something.
    for step in range(1, 9):
        assert f"\n{step}. " in out, f"step {step} missing from the output"

    # The two properties the example exists to demonstrate.
    assert "<- the future" in out, "no point-in-time leak was shown"
    assert "-> version 1" in out and "-> version 2" in out, (
        "the served version did not change with the alias"
    )
    assert "<- champion" in out, "the champion was not identified in the listing"

    # And it really wrote a feature set and a tracking store.
    assert (tmp_path / "mlflow.db").exists()
    assert (
        tmp_path
        / "offline"
        / example.FEATURE_SET
        / example.FEATURE_SET_VERSION
        / "data.parquet"
    ).exists()
