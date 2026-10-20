"""RQ2 selector contract tests: valid, positive-cost inputs only."""

import numpy as np
import pytest

from src.tcp_ci.rq2_selector import (
    risk_density_select,
    milp_select,
)


@pytest.mark.parametrize("rho", [0.90, 0.95, 0.99])
@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_risk_retention_constraint(selector, rho):
    result = selector(
        test_ids=["a", "b", "c", "d"],
        probabilities=[0.10, 0.40, 0.30, 0.20],
        costs_ms=[2.0, 3.0, 4.0, 1.0],
        rho=rho,
    )

    assert result.retained_risk + 1e-9 >= (
        rho * result.total_risk
    )
    assert set(result.selected_test_ids).issubset(
        {"a", "b", "c", "d"}
    )
    assert len(result.selected_test_ids) == len(
        set(result.selected_test_ids)
    )


def test_exact_solver_finds_cheaper_solution_than_greedy():
    data = dict(
        test_ids=["a", "b", "c"],
        probabilities=[0.60, 0.50, 0.50],
        costs_ms=[4.0, 3.6, 3.6],
        rho=0.50,
    )

    greedy = risk_density_select(**data)
    exact = milp_select(**data)

    assert exact.estimated_cost_ms <= (
        greedy.estimated_cost_ms + 1e-8
    )
    assert set(exact.selected_test_ids) == {"b", "c"}
    assert exact.estimated_cost_ms == pytest.approx(7.2)


@pytest.mark.parametrize(
    "invalid_costs",
    [
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, np.nan],
        [1.0, np.inf],
    ],
)
@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_nonpositive_or_nonfinite_costs(
    selector, invalid_costs
):
    with pytest.raises(ValueError):
        selector(
            test_ids=["a", "b"],
            probabilities=[0.4, 0.6],
            costs_ms=invalid_costs,
            rho=0.90,
        )


@pytest.mark.parametrize(
    "invalid_probabilities",
    [
        [-0.1, 0.5],
        [0.2, 1.1],
        [0.2, np.nan],
        [0.2, np.inf],
    ],
)
@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_invalid_probabilities(
    selector, invalid_probabilities
):
    with pytest.raises(ValueError):
        selector(
            test_ids=["a", "b"],
            probabilities=invalid_probabilities,
            costs_ms=[2.0, 3.0],
            rho=0.90,
        )


@pytest.mark.parametrize("rho", [-0.1, 0.0, 1.01])
@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_invalid_rho(selector, rho):
    with pytest.raises(ValueError):
        selector(
            test_ids=["a", "b"],
            probabilities=[0.4, 0.6],
            costs_ms=[2.0, 3.0],
            rho=rho,
        )


@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_duplicate_test_ids(selector):
    with pytest.raises(ValueError):
        selector(
            test_ids=["a", "a"],
            probabilities=[0.4, 0.6],
            costs_ms=[2.0, 3.0],
            rho=0.90,
        )


# --- WP3.4 selector correctness tests ---

from itertools import product
from types import SimpleNamespace

import src.tcp_ci.rq2_selector as selector_module


@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_empty_test_set(selector):
    with pytest.raises(ValueError):
        selector([], [], [], 0.90)


@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_zero_total_risk_explicitly_rejected(selector):
    # Temporary prototype behavior; NOT a frozen RQ2 policy.
    with pytest.raises(ValueError, match="Zero-total-risk"):
        selector(
            ["a", "b"],
            [0.0, 0.0],
            [2.0, 3.0],
            0.90,
        )


@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_rejects_mismatched_lengths(selector):
    with pytest.raises(ValueError):
        selector(
            ["a", "b"],
            [0.5],
            [2.0, 3.0],
            0.90,
        )


@pytest.mark.parametrize(
    "selector",
    [risk_density_select, milp_select],
)
def test_selection_repeatability(selector):
    args = dict(
        test_ids=["a", "b", "c"],
        probabilities=[0.2, 0.3, 0.5],
        costs_ms=[2.0, 3.0, 4.0],
        rho=0.90,
    )

    first = selector(**args)
    second = selector(**args)

    assert first.selected_test_ids == second.selected_test_ids
    assert first.estimated_cost_ms == second.estimated_cost_ms


@pytest.mark.parametrize(
    "probabilities,costs,rho",
    [
        ([0.1, 0.4, 0.3, 0.2],
         [2.0, 3.0, 4.0, 1.0], 0.90),
        ([0.6, 0.5, 0.5],
         [4.0, 3.6, 3.6], 0.50),
        ([0.25, 0.15, 0.1, 0.5],
         [6.0, 2.0, 2.0, 5.0], 0.95),
        ([0.1, 0.0, 0.9],
         [1.0, 3.0, 4.0], 1.00),
    ],
)
def test_milp_matches_bruteforce_optimum(
    probabilities, costs, rho
):
    n = len(probabilities)
    total_risk = sum(probabilities)
    required_risk = rho * total_risk

    feasible_costs = []

    for mask in product([0, 1], repeat=n):
        retained = sum(
            p for p, selected in zip(probabilities, mask)
            if selected
        )

        if retained + 1e-10 >= required_risk:
            cost = sum(
                c for c, selected in zip(costs, mask)
                if selected
            )
            feasible_costs.append(cost)

    assert feasible_costs

    result = milp_select(
        test_ids=[f"test_{i}" for i in range(n)],
        probabilities=probabilities,
        costs_ms=costs,
        rho=rho,
    )

    assert result.solver_status == "optimal"
    assert result.estimated_cost_ms == pytest.approx(
        min(feasible_costs)
    )
    assert result.retained_risk + 1e-10 >= required_risk


def test_milp_rejects_unproven_solver_result(monkeypatch):
    def fake_milp(**kwargs):
        return SimpleNamespace(
            status=1,
            x=None,
            success=False,
            message="Time limit reached",
        )

    monkeypatch.setattr(
        selector_module,
        "milp",
        fake_milp,
    )

    with pytest.raises(
        RuntimeError,
        match="did not prove optimality",
    ):
        milp_select(
            test_ids=["a", "b"],
            probabilities=[0.4, 0.6],
            costs_ms=[2.0, 3.0],
            rho=0.90,
        )


# --- WP3.5 deterministic tie-breaking ---


def test_risk_density_tie_independent_of_input_order():
    """Identical candidates must produce the same selected IDs."""

    first = risk_density_select(
        test_ids=["a", "b"],
        probabilities=[0.5, 0.5],
        costs_ms=[1.0, 1.0],
        rho=0.50,
    )

    reversed_input = risk_density_select(
        test_ids=["b", "a"],
        probabilities=[0.5, 0.5],
        costs_ms=[1.0, 1.0],
        rho=0.50,
    )

    assert first.selected_test_ids == (
        reversed_input.selected_test_ids
    )


# --- WP3.7 small-risk numerical regression ---


def test_milp_preserves_very_small_predicted_risk():
    result = milp_select(
        test_ids=["a", "b", "c"],
        probabilities=[1e-10, 2e-10, 3e-10],
        costs_ms=[1.0, 2.0, 3.0],
        rho=0.95,
    )

    assert result.solver_status == "optimal"
    assert set(result.selected_test_ids) == {"a", "b", "c"}
    assert result.retained_risk >= (
        0.95 * result.total_risk
    )
    assert result.estimated_cost_ms == pytest.approx(6.0)

# --- RQ2 greedy linear accumulation regression ---


def test_risk_density_does_not_recompute_every_prefix(monkeypatch):
    original_fsum = selector_module.fsum
    calls = []

    def counted_fsum(values):
        calls.append(1)
        return original_fsum(values)

    monkeypatch.setattr(
        selector_module, "fsum", counted_fsum
    )

    n = 200
    result = risk_density_select(
        test_ids=[f"t{i}" for i in range(n)],
        probabilities=[0.5] * n,
        costs_ms=[1.0] * n,
        rho=1.0,
    )

    assert len(result.selected_test_ids) == n

    # Input validation and result validation may use fsum.
    # The greedy loop must not recompute every prefix.
    assert len(calls) <= 4
