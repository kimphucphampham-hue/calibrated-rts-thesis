"""Automated tests for candidate M2-06 orchestration.

These tests do not approve or freeze M2-06.
"""

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci import rq2_cold_start_orchestrator as orchestration


def make_target(costs):
    return pd.DataFrame(
        {
            "project_id": ["project-A", "project-A"],
            "build_index": [10, 10],
            "test_id": ["test-a", "test-b"],
            "execution_time_mean_history_ms": costs,
            "execution_time_ms": [999999.0, 999999.0],
            "failed": [0, 1],
        },
        index=[20, 21],
    )


def make_history(positive=False):
    return pd.DataFrame(
        {
            "project_id": ["project-A"],
            "build_index": [9],
            "execution_time_ms": [
                20.0 if positive else 0.0
            ],
        }
    )


def run(target, history, probabilities=(0.9, 0.1), rho=0.8):
    return orchestration.select_with_cold_start_fallback(
        target=target,
        history=history,
        probabilities=probabilities,
        rho=rho,
    )


def test_normal_selection_uses_risk_density():
    result = run(
        make_target([5.0, 100.0]),
        make_history(positive=False),
    )

    assert result.selection_mode == "normal_risk_density"
    assert result.fallback_reason is None
    assert set(result.selected_test_ids) == {"test-a"}
    assert result.n_candidates == 2
    assert result.estimated_selected_cost_ms == pytest.approx(
        5.0
    )
    assert result.retained_risk_ratio >= 0.8


def test_missing_and_zero_without_history_select_all():
    result = run(
        make_target([0.0, np.nan]),
        make_history(positive=False),
    )

    assert result.selection_mode == "fallback_retest_all"
    assert result.fallback_reason == (
        "no_positive_prior_cost_history"
    )
    assert result.selected_test_ids == (
        "test-a",
        "test-b",
    )
    assert result.estimated_selected_cost_ms is None
    assert result.retained_risk_ratio == pytest.approx(1.0)

    diagnostic = result.diagnostic_record()

    assert diagnostic["n_candidates"] == 2
    assert diagnostic["n_selected"] == 2
    assert diagnostic["selection_mode"] == (
        "fallback_retest_all"
    )
    assert diagnostic["estimated_selected_cost_ms"] is None


def test_earlier_positive_history_uses_median_not_fallback():
    target = make_target([0.0, np.nan])

    history = pd.DataFrame(
        {
            "project_id": ["project-A", "project-A"],
            "build_index": [7, 8],
            "execution_time_ms": [10.0, 30.0],
        }
    )

    result = run(
        target,
        history,
        rho=0.95,
    )

    assert result.selection_mode == "normal_risk_density"
    assert result.fallback_reason is None
    assert set(result.selected_test_ids) == {
        "test-a", "test-b"
    }

    # Earlier positive median = 20 ms.
    assert result.estimated_selected_cost_ms == pytest.approx(
        40.0
    )


def test_current_future_and_other_project_cannot_rescue_cold_start():
    history = pd.DataFrame(
        {
            "project_id": [
                "project-A",
                "project-A",
                "project-B",
            ],
            "build_index": [10, 11, 1],
            "execution_time_ms": [
                10000.0, 20000.0, 30000.0
            ],
        }
    )

    result = run(
        make_target([0.0, np.nan]),
        history,
    )

    assert result.selection_mode == "fallback_retest_all"
    assert len(result.selected_test_ids) == 2


def test_target_outcomes_do_not_influence_fallback():
    original = make_target([0.0, np.nan])

    modified = original.copy()
    modified["execution_time_ms"] = [
        100000000.0,
        1.0,
    ]
    modified["failed"] = [1, 0]

    first = run(original, make_history())
    second = run(modified, make_history())

    assert first == second


def test_invalid_historical_cost_does_not_trigger_fallback():
    target = make_target([-1.0, np.nan])

    with pytest.raises(
        ValueError,
        match="Negative historical estimated cost",
    ):
        run(target, make_history())


def test_invalid_history_does_not_trigger_fallback():
    history = pd.DataFrame(
        {
            "project_id": ["project-A"],
            "build_index": [9],
            "execution_time_ms": [-5.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Negative historical realized duration",
    ):
        run(
            make_target([0.0, np.nan]),
            history,
        )


def test_unrelated_value_error_is_not_swallowed(monkeypatch):
    def unrelated_error(*args, **kwargs):
        raise ValueError("Unrelated implementation error.")

    monkeypatch.setattr(
        orchestration,
        "estimate_build_costs",
        unrelated_error,
    )

    with pytest.raises(
        ValueError,
        match="Unrelated implementation error",
    ):
        run(
            make_target([0.0, np.nan]),
            make_history(),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [np.nan, 0.2],
        [-0.1, 0.5],
        [1.1, 0.2],
    ],
)
def test_invalid_probabilities_cannot_trigger_fallback(
    probabilities,
):
    with pytest.raises(
        ValueError,
        match="Probabilities must be finite",
    ):
        run(
            make_target([0.0, np.nan]),
            make_history(),
            probabilities=probabilities,
        )


def test_probability_row_alignment_is_checked():
    target = make_target([0.0, np.nan])

    probabilities = pd.Series(
        [0.9, 0.1],
        index=[21, 20],
    )

    with pytest.raises(
        ValueError,
        match="index mismatch",
    ):
        run(
            target,
            make_history(),
            probabilities=probabilities,
        )


def test_zero_total_risk_remains_m2_02_decision():
    with pytest.raises(
        ValueError,
        match="M2-02 is pending",
    ):
        run(
            make_target([0.0, np.nan]),
            make_history(),
            probabilities=[0.0, 0.0],
        )


@pytest.mark.parametrize("rho", [0.0, 1.1, np.nan])
def test_invalid_rho_cannot_trigger_fallback(rho):
    with pytest.raises(
        ValueError,
        match="rho must be finite",
    ):
        run(
            make_target([0.0, np.nan]),
            make_history(),
            rho=rho,
        )


def test_duplicate_test_ids_are_rejected():
    target = make_target([0.0, np.nan])
    target["test_id"] = ["same", "same"]

    with pytest.raises(
        ValueError,
        match="Duplicate test_id",
    ):
        run(
            target,
            make_history(),
        )
