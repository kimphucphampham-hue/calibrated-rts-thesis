"""Tests for the RQ2 M2-01 extended sensitivity pilot.

These tests validate prototype behavior only.
They do not constitute M2 Method Freeze.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_rq2_cost_sensitivity_extended as extended


def make_rows():
    return pd.DataFrame(
        {
            "project_id": ["project-a"] * 3,
            "build_id": ["b2"] * 3,
            "build_index": [2] * 3,
            "test_id": ["a", "b", "c"],
            "p_platt": [0.6, 0.3, 0.1],
            "failed": [1, 0, 1],
            "execution_time_ms": [5.0, 0.0, 15.0],
            "execution_time_mean_history_ms": [
                8.0, 0.0, 20.0
            ],
        },
        index=[10, 11, 12],
    )


def make_costs():
    return pd.DataFrame(
        {"estimated_cost_ms": [2.0, 1.0, 3.0]},
        index=[10, 11, 12],
    )


def calculate_result(rows=None, costs=None, selected=None):
    if rows is None:
        rows = make_rows()

    if costs is None:
        costs = make_costs()

    if selected is None:
        selected = ["a", "b"]

    return extended.make_result(
        project="project-a",
        build_index=2,
        rows=rows,
        costs=costs,
        selection=SimpleNamespace(
            selected_test_ids=selected
        ),
        rho=0.90,
        policy="positive_floor",
        floor_ms=1.0,
    )


def make_policy_pair():
    base = {
        "project": "project-a",
        "build_index": 2,
        "rho": 0.95,
        "n_zero_history": 1,
        "total_failures": 2,
        "n_selected": 2,
        "tsr": 2 / 3,
        "rtr": 0.5,
        "missed_failures": 1,
    }

    return pd.DataFrame(
        [
            {
                **base,
                "policy": "positive_floor",
                "selected_test_ids": "a|b",
                "estimated_selected_ms": 10.0,
            },
            {
                **base,
                "policy": "historical_median",
                "selected_test_ids": "b|c",
                "estimated_selected_ms": 30.0,
            },
        ]
    )


def test_chronological_sampling_ignores_failure_outcomes():
    first = pd.DataFrame(
        {
            "build_index": range(1, 31),
            "failed": [0] * 30,
        }
    )

    second = first.copy()
    second["failed"] = 1

    sampled_first = extended.choose_builds(first)
    sampled_second = extended.choose_builds(second)

    assert sampled_first == sampled_second
    assert len(sampled_first) == 12
    assert sampled_first[0] == 1
    assert sampled_first[-1] == 30
    assert sampled_first == sorted(set(sampled_first))


def test_pair_comparison_detects_different_test_ids():
    pairs = extended.build_pairs(make_policy_pair())

    assert len(pairs) == 1

    result = pairs.iloc[0]

    assert not bool(result["same_selected_ids"])
    assert result["jaccard"] == pytest.approx(1 / 3)
    assert result["estimated_cost_delta_ms"] == pytest.approx(
        -20.0
    )


def test_pair_comparison_rejects_missing_policy():
    incomplete = make_policy_pair().iloc[[0]].copy()

    with pytest.raises(
        ValueError,
        match="Incomplete paired comparison",
    ):
        extended.build_pairs(incomplete)


def test_metrics_use_realized_runtime_and_failure_labels():
    result = calculate_result()

    assert result["n_tests"] == 3
    assert result["n_selected"] == 2
    assert result["n_zero_history"] == 1
    assert result["n_selected_zero_history"] == 1

    assert result["tsr"] == pytest.approx(2 / 3)
    assert result["rtr"] == pytest.approx(0.25)

    assert result["observed_full_ms"] == pytest.approx(20.0)
    assert result["observed_selected_ms"] == pytest.approx(5.0)
    assert result["estimated_selected_ms"] == pytest.approx(3.0)

    assert result["total_failures"] == 2
    assert result["captured_failures"] == 1
    assert result["missed_failures"] == 1
    assert result["failure_recall"] == pytest.approx(0.5)
    assert result["missed_failure_rate"] == pytest.approx(0.5)

    assert result["retained_risk_ratio"] == pytest.approx(
        0.90
    )


def test_result_rejects_misaligned_cost_rows():
    costs = make_costs().reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="alignment mismatch",
    ):
        calculate_result(costs=costs)


def test_result_rejects_risk_coverage_shortfall():
    with pytest.raises(
        ValueError,
        match="Risk coverage constraint violated",
    ):
        calculate_result(selected=["a"])


def make_synthetic_project():
    train = pd.DataFrame(
        {
            "project_id": ["project-a", "project-a"],
            "build_index": [1, 1],
            "execution_time_ms": [30.0, 90.0],
        }
    )

    aligned = pd.DataFrame(
        {
            "project_id": ["project-a", "project-a"],
            "build_id": ["b2", "b2"],
            "build_index": [2, 2],
            "test_id": ["a", "b"],
            "p_platt": [0.6, 0.4],
            "failed": [0, 1],
            "execution_time_ms": [0.0, 25.0],
            "execution_time_mean_history_ms": [
                0.0, 10.0
            ],
        }
    )

    test = pd.concat(
        [
            aligned[
                extended.HISTORY_COLUMNS
            ],
            pd.DataFrame(
                {
                    "project_id": ["project-a"],
                    "build_index": [3],
                    "execution_time_ms": [1000000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    return train, test, aligned, 1.0


def test_integration_ignores_future_build_durations(
    monkeypatch,
):
    data = make_synthetic_project()

    monkeypatch.setattr(
        extended,
        "load_project",
        lambda project: data,
    )

    monkeypatch.setattr(
        extended,
        "RHOS",
        (0.99,),
    )

    results, skipped = extended.run_project(
        "project-a"
    )

    assert skipped == []
    assert len(results) == 2

    by_policy = {
        row["policy"]: row
        for row in results
    }

    # Both tests must be selected at rho=0.99.
    # Prior positive durations are 30 and 90 ms:
    # historical median = 60 ms.
    # Future build 3 contains 1,000,000 ms and
    # must have no influence on build 2.
    assert by_policy[
        "historical_median"
    ]["estimated_selected_ms"] == pytest.approx(
        60.0 + 10.0
    )

    assert by_policy[
        "positive_floor"
    ]["estimated_selected_ms"] == pytest.approx(
        1.0 + 10.0
    )

    assert all(
        row["retained_risk_ratio"] >= 0.99
        for row in results
    )


def test_zero_total_risk_is_explicitly_skipped(
    monkeypatch,
):
    train, test, aligned, floor = (
        make_synthetic_project()
    )

    aligned = aligned.copy()
    aligned["p_platt"] = 0.0

    monkeypatch.setattr(
        extended,
        "load_project",
        lambda project: (
            train, test, aligned, floor
        ),
    )

    results, skipped = extended.run_project(
        "project-a"
    )

    assert results == []
    assert len(skipped) == 1
    assert skipped[0]["build_index"] == 2
    assert skipped[0]["reason"] == (
        "zero_total_predicted_risk"
    )
