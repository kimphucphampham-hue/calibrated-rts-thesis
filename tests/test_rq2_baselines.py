"""RQ2 baseline tests — test-first implementation."""

import pytest

from src.tcp_ci.rq2_baselines import (
    retest_all,
    random_matched_count,
)


def test_retest_all_selects_every_test():
    result = retest_all(
        test_ids=["a", "b", "c"],
        costs_ms=[10.0, 20.0, 30.0],
    )

    assert set(result.selected_test_ids) == {
        "a", "b", "c"
    }
    assert result.estimated_cost_ms == pytest.approx(60.0)
    assert result.baseline_name == "retest_all"


def test_retest_all_independent_of_input_order():
    first = retest_all(
        ["a", "b", "c"], [1.0, 2.0, 3.0]
    )

    second = retest_all(
        ["c", "a", "b"], [3.0, 1.0, 2.0]
    )

    assert first.selected_test_ids == second.selected_test_ids


def test_random_matches_requested_count():
    result = random_matched_count(
        test_ids=["a", "b", "c", "d"],
        costs_ms=[10.0, 20.0, 30.0, 40.0],
        k=2,
        seed=42,
    )

    assert len(result.selected_test_ids) == 2
    assert len(set(result.selected_test_ids)) == 2
    assert result.baseline_name == "random_matched_count"


def test_random_is_reproducible():
    args = {
        "test_ids": ["a", "b", "c", "d", "e"],
        "costs_ms": [1.0] * 5,
        "k": 3,
        "seed": 42,
    }

    first = random_matched_count(**args)
    second = random_matched_count(**args)

    assert first == second


def test_random_independent_of_input_order():
    first = random_matched_count(
        ["a", "b", "c", "d"],
        [1.0, 2.0, 3.0, 4.0],
        k=2,
        seed=42,
    )

    second = random_matched_count(
        ["d", "b", "a", "c"],
        [4.0, 2.0, 1.0, 3.0],
        k=2,
        seed=42,
    )

    assert first == second


@pytest.mark.parametrize("k", [0, 4])
def test_random_rejects_invalid_count(k):
    with pytest.raises(ValueError):
        random_matched_count(
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            k=k,
            seed=42,
        )


@pytest.mark.parametrize(
    "selector",
    [retest_all, random_matched_count],
)
def test_baselines_reject_duplicate_ids(selector):
    args = {
        "test_ids": ["a", "a"],
        "costs_ms": [1.0, 2.0],
    }

    if selector is random_matched_count:
        args.update(k=1, seed=42)

    with pytest.raises(ValueError):
        selector(**args)


@pytest.mark.parametrize(
    "selector",
    [retest_all, random_matched_count],
)
def test_baselines_reject_invalid_cost(selector):
    args = {
        "test_ids": ["a", "b"],
        "costs_ms": [1.0, float("nan")],
    }

    if selector is random_matched_count:
        args.update(k=1, seed=42)

    with pytest.raises(ValueError):
        selector(**args)

# --- WP4.4 budget-capped random baseline ---


def _budget_select(*args, **kwargs):
    from src.tcp_ci.rq2_baselines import random_matched_budget
    return random_matched_budget(*args, **kwargs)


def test_random_budget_never_exceeds_cap():
    result = _budget_select(
        test_ids=["a", "b", "c", "d"],
        costs_ms=[10.0, 20.0, 30.0, 40.0],
        budget_ms=50.0,
        seed=42,
    )

    assert result.estimated_cost_ms <= 50.0
    assert len(result.selected_test_ids) == len(
        set(result.selected_test_ids)
    )
    assert set(result.selected_test_ids) <= {
        "a", "b", "c", "d"
    }
    assert result.baseline_name == "random_matched_budget"


def test_random_budget_is_reproducible():
    args = {
        "test_ids": ["a", "b", "c", "d"],
        "costs_ms": [10.0, 20.0, 30.0, 40.0],
        "budget_ms": 50.0,
        "seed": 42,
    }

    assert _budget_select(**args) == _budget_select(**args)


def test_random_budget_independent_of_input_order():
    first = _budget_select(
        ["a", "b", "c", "d"],
        [10.0, 20.0, 30.0, 40.0],
        budget_ms=50.0,
        seed=42,
    )

    second = _budget_select(
        ["d", "b", "a", "c"],
        [40.0, 20.0, 10.0, 30.0],
        budget_ms=50.0,
        seed=42,
    )

    assert first == second


def test_random_budget_skips_unaffordable_test():
    result = _budget_select(
        test_ids=["expensive", "cheap"],
        costs_ms=[100.0, 2.0],
        budget_ms=2.0,
        seed=42,
    )

    assert result.selected_test_ids == ("cheap",)
    assert result.estimated_cost_ms == pytest.approx(2.0)


def test_random_budget_can_return_empty_selection():
    result = _budget_select(
        test_ids=["a", "b"],
        costs_ms=[3.0, 4.0],
        budget_ms=2.0,
        seed=42,
    )

    assert result.selected_test_ids == ()
    assert result.estimated_cost_ms == pytest.approx(0.0)


@pytest.mark.parametrize(
    "invalid_budget",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_random_budget_rejects_invalid_limit(invalid_budget):
    with pytest.raises(ValueError):
        _budget_select(
            test_ids=["a", "b"],
            costs_ms=[1.0, 2.0],
            budget_ms=invalid_budget,
            seed=42,
        )


# --- WP4.5 historical selection baselines ---


def _history_selector(name, *args, **kwargs):
    from src.tcp_ci import rq2_baselines

    return getattr(rq2_baselines, name)(*args, **kwargs)


def test_failure_history_prefers_higher_failure_rate():
    result = _history_selector(
        "failure_history_select",
        test_ids=["a", "b", "c"],
        costs_ms=[10.0, 10.0, 10.0],
        failure_rates=[0.1, 0.8, 0.4],
        k=2,
    )

    assert result.selected_test_ids == ("b", "c")
    assert result.baseline_name == "failure_history"


def test_history_duration_prefers_risk_per_cost():
    result = _history_selector(
        "history_duration_select",
        test_ids=["a", "b"],
        costs_ms=[10.0, 1.0],
        failure_rates=[0.9, 0.5],
        k=1,
    )

    assert result.selected_test_ids == ("b",)
    assert result.estimated_cost_ms == pytest.approx(1.0)
    assert result.baseline_name == "history_duration"


@pytest.mark.parametrize(
    "name",
    ["failure_history_select", "history_duration_select"],
)
def test_historical_baseline_matches_requested_count(name):
    result = _history_selector(
        name,
        test_ids=["a", "b", "c", "d"],
        costs_ms=[1.0, 2.0, 3.0, 4.0],
        failure_rates=[0.4, 0.3, 0.2, 0.1],
        k=2,
    )

    assert len(result.selected_test_ids) == 2
    assert len(set(result.selected_test_ids)) == 2


@pytest.mark.parametrize(
    "name",
    ["failure_history_select", "history_duration_select"],
)
def test_historical_ties_independent_of_input_order(name):
    first = _history_selector(
        name,
        ["b", "a"],
        [2.0, 2.0],
        [0.5, 0.5],
        k=1,
    )

    second = _history_selector(
        name,
        ["a", "b"],
        [2.0, 2.0],
        [0.5, 0.5],
        k=1,
    )

    assert first.selected_test_ids == ("a",)
    assert first == second


@pytest.mark.parametrize(
    "name",
    ["failure_history_select", "history_duration_select"],
)
def test_historical_baseline_rejects_missing_rate(name):
    with pytest.raises(ValueError):
        _history_selector(
            name,
            test_ids=["a", "b"],
            costs_ms=[1.0, 2.0],
            failure_rates=[0.5, float("nan")],
            k=1,
        )

def test_random_budget_handles_large_finite_costs():
    result = _budget_select(
        test_ids=["a", "b"],
        costs_ms=[1e308, 1e308],
        budget_ms=1e308,
        seed=42,
    )

    assert len(result.selected_test_ids) == 1
    assert result.estimated_cost_ms == 1e308


def test_history_duration_avoids_ratio_overflow():
    result = _history_selector(
        "history_duration_select",
        test_ids=["a", "z"],
        costs_ms=[2e-320, 1e-320],
        failure_rates=[0.5, 0.5],
        k=1,
    )

    assert result.selected_test_ids == ("z",)