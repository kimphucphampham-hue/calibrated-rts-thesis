"""Synthetic-only, pre-freeze H2 budget-comparable baseline pairing tests."""

import pytest

from src.tcp_ci.rq2_h2_budget_pairing import pair_h2_budget_comparators_for_build

CONDITION = (
    "model=lr",
    "probability=platt",
    "rho=0.95",
    "cost_policy=positive_median",
    "observation=retrospective_full_information",
)


def _run(*, ids=None, costs=None, rates=None, reference=None, condition=CONDITION):
    return pair_h2_budget_comparators_for_build(
        project_id="datasets/Example@repo",
        build_index=12,
        condition_key=condition,
        test_ids=["a", "b", "c", "d"] if ids is None else ids,
        estimated_costs_ms=[4.0, 2.0, 7.0, 3.0] if costs is None else costs,
        historical_failure_rates=[0.8, 0.6, 0.99, 0.4] if rates is None else rates,
        reference_selected_test_ids=["a", "b"] if reference is None else reference,
    )


def test_same_estimated_budget_for_random_and_both_history_baselines():
    result = _run()
    assert result.random_pair.reference_estimated_cost_ms == 6.0
    assert len(result.random_pair.random_replicates) == 30
    assert len(result.random_budget_utilizations) == 30
    for rep in result.random_pair.random_replicates:
        assert rep.budget_capped is not None
        assert rep.budget_capped.estimated_cost_ms <= 6.0
        assert rep.budget_seed is not None
    assert result.failure_history_budget.selected_test_ids == ("a", "b")
    assert result.history_duration_budget.selected_test_ids == ("b", "a")
    assert result.failure_history_budget.estimated_cost_ms == 6.0
    assert result.history_duration_budget.estimated_cost_ms == 6.0
    assert result.failure_history_budget_utilization == 1.0
    assert result.history_duration_budget_utilization == 1.0
    assert all(0 <= value <= 1.0 for value in result.random_budget_utilizations)
    assert result.h2_primary_rho is True


def test_input_row_permutation_changes_neither_selection_nor_seeds():
    a = _run()
    b = _run(ids=["d", "b", "a", "c"], costs=[3.0, 2.0, 4.0, 7.0],
             rates=[0.4, 0.6, 0.8, 0.99], reference=["b", "a"])
    assert a.failure_history_budget == b.failure_history_budget
    assert a.history_duration_budget == b.history_duration_budget
    assert a.random_pair == b.random_pair
    assert a.random_budget_utilizations == b.random_budget_utilizations


def test_unaffordable_candidates_are_skipped_and_underspend_visible():
    result = _run(reference=["a"])
    assert result.random_pair.reference_estimated_cost_ms == 4.0
    assert result.failure_history_budget.selected_test_ids == ("a",)
    assert result.history_duration_budget.selected_test_ids == ("b",)
    assert result.history_duration_budget.estimated_cost_ms == 2.0
    assert result.history_duration_budget_utilization == 0.5


def test_zero_historical_rates_are_allowed_and_ties_deterministic():
    result = _run(rates=[0.0, 0.0, 0.0, 0.0])
    assert result.failure_history_budget.selected_test_ids == ("a", "b")
    assert result.history_duration_budget.selected_test_ids == ("a", "b")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.1])
def test_invalid_history_rates_fail_closed(bad):
    with pytest.raises(ValueError):
        _run(rates=[bad, 0.6, 0.99, 0.4])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_estimated_costs_fail_closed(bad):
    with pytest.raises((ValueError, TypeError)):
        _run(costs=[bad, 2.0, 7.0, 3.0])


def test_h2_inference_label_only_for_rho_at_least_95():
    label_90 = CONDITION[:2] + ("rho=0.90",) + CONDITION[3:]
    label_99 = CONDITION[:2] + ("rho=0.99",) + CONDITION[3:]
    assert _run(condition=label_90).h2_primary_rho is False
    assert _run(condition=label_99).h2_primary_rho is True


def test_noncanonical_rho_is_rejected():
    label = CONDITION[:2] + ("rho=0.950",) + CONDITION[3:]
    with pytest.raises(ValueError, match="rho"):
        _run(condition=label)


def test_target_oracle_is_not_in_selection_api():
    with pytest.raises(TypeError):
        pair_h2_budget_comparators_for_build(
            project_id="p", build_index=1, condition_key=CONDITION,
            test_ids=["a"], estimated_costs_ms=[1.0],
            historical_failure_rates=[0.0], reference_selected_test_ids=["a"],
            failed=[1], execution_time_ms=[999999.0],
        )
