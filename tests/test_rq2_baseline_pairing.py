"""Pre-freeze one-build M2-04/M2-05 deterministic pairing contract tests."""

import inspect

import pytest

from src.tcp_ci.rq2_baseline_pairing import (
    WORKING_REPETITIONS,
    derive_random_seed,
    pair_random_baselines_for_build,
)

CONDITION = (
    "model=logistic_regression", "probability=platt", "rho=0.95",
    "cost_policy=historical_median", "observation=retrospective_full_information",
)


def _run(*, reverse=False, with_budget=True):
    ids = ("c", "a", "d", "b")
    costs = (30.0, 10.0, 40.0, 20.0)
    if reverse:
        ids, costs = ids[::-1], costs[::-1]
    return pair_random_baselines_for_build(
        project_id="project-A", build_index=12, condition_key=CONDITION,
        test_ids=ids, estimated_costs_ms=costs,
        reference_selected_test_ids=("d", "a") if reverse else ("a", "d"),
        include_budget_sensitivity=with_budget,
    )


def test_30_repetitions_match_reference_selected_count_and_budget_separate():
    result = _run()
    assert WORKING_REPETITIONS == 30
    assert len(result.random_replicates) == 30
    assert result.reference_selected_test_ids == ("a", "d")
    assert result.matched_count_k == 2
    assert result.reference_estimated_cost_ms == 50.0
    assert result.full_retest.selected_test_ids == ("a", "b", "c", "d")
    assert result.full_retest.estimated_cost_ms == 100.0
    assert [rep.repetition for rep in result.random_replicates] == list(range(30))
    for rep in result.random_replicates:
        assert len(rep.matched_count.selected_test_ids) == 2
        assert rep.matched_count.baseline_name == "random_matched_count"
        assert rep.budget_capped.baseline_name == "random_matched_budget"
        assert rep.budget_capped.estimated_cost_ms <= 50.0
        assert rep.budget_seed is not None
        assert rep.budget_seed != rep.matched_count_seed


def test_reproducible_and_independent_of_input_row_and_reference_order():
    assert _run() == _run(reverse=True)
    assert _run() == _run()
    seeds = [rep.matched_count_seed for rep in _run().random_replicates]
    assert len(set(seeds)) == 30


def test_seed_scheme_golden_vector_and_key_separation():
    args = dict(project_id="project-A", build_index=12, condition_key=CONDITION,
                repetition=0, comparator="random_matched_count")
    assert derive_random_seed(**args) == 4968361439849426895
    assert derive_random_seed(**{**args, "project_id": "project-B"}) != derive_random_seed(**args)
    assert derive_random_seed(**{**args, "build_index": 13}) != derive_random_seed(**args)
    assert derive_random_seed(**{**args, "repetition": 1}) != derive_random_seed(**args)
    assert derive_random_seed(**{**args, "comparator": "random_matched_budget"}) != derive_random_seed(**args)
    altered = (*CONDITION[:-1], "observation=another_regime")
    assert derive_random_seed(**{**args, "condition_key": altered}) != derive_random_seed(**args)


def test_budget_sensitivity_is_explicitly_optional():
    result = _run(with_budget=False)
    assert all(rep.budget_capped is None and rep.budget_seed is None
               for rep in result.random_replicates)


@pytest.mark.parametrize("selected", [(), ("not-a-candidate",), ("a", "a")])
def test_invalid_reference_selection_fails_closed(selected):
    with pytest.raises(ValueError, match="Reference selection"):
        pair_random_baselines_for_build(
            project_id="project-A", build_index=12, condition_key=CONDITION,
            test_ids=("a", "b"), estimated_costs_ms=(1., 2.),
            reference_selected_test_ids=selected,
        )


@pytest.mark.parametrize("bad_condition", [
    ("model=x",),
    ("probability=x", "model=x", "rho=0.9", "cost_policy=x", "observation=x"),
    ("model=", "probability=x", "rho=0.9", "cost_policy=x", "observation=x"),
])
def test_invalid_condition_key_fails_closed(bad_condition):
    with pytest.raises(ValueError, match="condition_key"):
        pair_random_baselines_for_build(
            project_id="project-A", build_index=12, condition_key=bad_condition,
            test_ids=("a", "b"), estimated_costs_ms=(1., 2.),
            reference_selected_test_ids=("a",),
        )


@pytest.mark.parametrize("bad_seed", [True, -1, 1.5])
def test_invalid_base_seed_rejected(bad_seed):
    with pytest.raises(ValueError, match="base_seed"):
        derive_random_seed(project_id="p", build_index=1,
                           condition_key=CONDITION, repetition=0,
                           comparator="random_matched_count", base_seed=bad_seed)


def test_invalid_candidates_costs_and_indexes_rejected():
    kwargs = dict(project_id="p", build_index=1, condition_key=CONDITION,
                  test_ids=("a", "b"), estimated_costs_ms=(1., 2.),
                  reference_selected_test_ids=("a",))
    with pytest.raises(ValueError):
        pair_random_baselines_for_build(**{**kwargs, "test_ids": ("a", "a")})
    with pytest.raises(ValueError):
        pair_random_baselines_for_build(**{**kwargs, "estimated_costs_ms": (float("nan"), 2.)})
    with pytest.raises(ValueError, match="build_index"):
        pair_random_baselines_for_build(**{**kwargs, "build_index": 0})


def test_selector_api_does_not_accept_oracle_or_target_outcomes():
    names = set(inspect.signature(pair_random_baselines_for_build).parameters)
    assert not names.intersection({"target_oracle", "failed", "execution_time_ms", "target_outcomes"})
    with pytest.raises(TypeError):
        pair_random_baselines_for_build(
            project_id="p", build_index=1, condition_key=CONDITION,
            test_ids=("a",), estimated_costs_ms=(1.,),
            reference_selected_test_ids=("a",), target_oracle=object(),
        )
