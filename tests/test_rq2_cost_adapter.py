"""Tests for the proposed RQ2 historical cost adapter."""

import pandas as pd
import pytest

from src.tcp_ci.rq2_cost_adapter import estimate_build_costs


def make_target(costs):
    return pd.DataFrame({
        "project_id": ["project_a"] * len(costs),
        "build_index": [10] * len(costs),
        "test_id": [f"test_{i}" for i in range(len(costs))],
        "execution_time_mean_history_ms": costs,
    })


def make_history(builds, durations):
    return pd.DataFrame({
        "project_id": ["project_a"] * len(builds),
        "build_index": builds,
        "execution_time_ms": durations,
    })


def test_preserves_positive_historical_cost():
    target = make_target([15.0])
    history = make_history([1, 2], [5.0, 25.0])

    result = estimate_build_costs(target, history)

    assert result["estimated_cost_ms"].tolist() == [15.0]


def test_missing_cost_uses_earlier_project_median():
    target = make_target([float("nan")])
    history = make_history([1, 2, 3], [10.0, 30.0, 50.0])

    result = estimate_build_costs(target, history)

    assert result["estimated_cost_ms"].tolist() == [30.0]


def test_zero_cost_uses_positive_history():
    target = make_target([0.0])
    history = make_history([1, 2, 3], [0.0, 10.0, 30.0])

    result = estimate_build_costs(target, history)

    assert result["estimated_cost_ms"].tolist() == [20.0]


def test_target_and_future_builds_are_excluded():
    target = make_target([float("nan")])
    history = make_history(
        [1, 2, 10, 11],
        [10.0, 30.0, 1000.0, 2000.0],
    )

    result = estimate_build_costs(target, history)

    assert result["estimated_cost_ms"].tolist() == [20.0]


def test_no_eligible_history_fails_explicitly():
    target = make_target([float("nan")])
    history = make_history([10, 11], [100.0, 200.0])

    with pytest.raises(ValueError):
        estimate_build_costs(target, history)


@pytest.mark.parametrize("bad_cost", [-1.0, float("inf")])
def test_invalid_historical_cost_rejected(bad_cost):
    target = make_target([bad_cost])
    history = make_history([1], [20.0])

    with pytest.raises(ValueError):
        estimate_build_costs(target, history)

# --- WP2.5 cost adapter integrity tests ---


def test_multiple_tests_in_one_build_preserve_order():
    target = make_target([15.0, float("nan"), 0.0])

    history = make_history(
        [1, 2, 10, 11],
        [10.0, 30.0, 1000.0, 2000.0],
    )

    # Other-project historical duration must be ignored.
    other = make_history([1], [9000.0])
    other["project_id"] = "project_b"

    history = pd.concat(
        [history, other],
        ignore_index=True,
    )

    result = estimate_build_costs(target, history)

    assert result["test_id"].tolist() == [
        "test_0", "test_1", "test_2"
    ]
    assert result["estimated_cost_ms"].tolist() == [
        15.0, 20.0, 20.0
    ]


def test_other_project_history_cannot_supply_fallback():
    target = make_target([float("nan")])

    history = make_history([1, 2], [10.0, 30.0])
    history["project_id"] = "project_b"

    with pytest.raises(
        ValueError,
        match="No eligible positive project duration",
    ):
        estimate_build_costs(target, history)
