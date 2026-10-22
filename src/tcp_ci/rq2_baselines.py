"""RQ2 selection baselines — pre-M2 prototype.

Decision-time inputs only. No target-build outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    selected_test_ids: tuple[str, ...]
    estimated_cost_ms: float
    baseline_name: str


def _validate_candidates(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
) -> tuple[list[str], np.ndarray]:
    ids = list(test_ids)

    if not ids:
        raise ValueError("Candidate set must not be empty.")

    if not all(isinstance(i, str) for i in ids):
        raise ValueError("Test IDs must be strings.")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate test IDs.")

    costs = np.asarray(costs_ms, dtype=float)

    if costs.ndim != 1 or len(costs) != len(ids):
        raise ValueError("Cost shape or length mismatch.")

    if not np.isfinite(costs).all():
        raise ValueError("Costs must be finite.")

    if (costs <= 0).any():
        raise ValueError("Costs must be positive.")

    return ids, costs


def retest_all(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
) -> BaselineResult:
    """Run every candidate test, independent of input order."""
    ids, costs = _validate_candidates(test_ids, costs_ms)

    order = sorted(range(len(ids)), key=lambda i: ids[i])

    return BaselineResult(
        selected_test_ids=tuple(ids[i] for i in order),
        estimated_cost_ms=fsum(
            float(costs[i]) for i in order
        ),
        baseline_name="retest_all",
    )


def random_matched_count(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
    k: int,
    seed: int,
) -> BaselineResult:
    """Uniform sample of exactly k tests without replacement.

    Input rows are canonicalized by test_id first.
    The seed must be supplied explicitly.
    """
    ids, costs = _validate_candidates(test_ids, costs_ms)

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or not (1 <= k <= len(ids))
    ):
        raise ValueError(
            "k must be an integer within [1, candidate count]."
        )

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError(
            "seed must be a nonnegative integer."
        )

    # Canonical order makes sampling independent of CSV row order.
    canonical_order = sorted(
        range(len(ids)),
        key=lambda i: ids[i],
    )

    rng = np.random.default_rng(int(seed))

    sampled_positions = rng.choice(
        len(ids),
        size=int(k),
        replace=False,
    )

    selected_indices = sorted(
        (canonical_order[int(pos)] for pos in sampled_positions),
        key=lambda i: ids[i],
    )

    return BaselineResult(
        selected_test_ids=tuple(
            ids[i] for i in selected_indices
        ),
        estimated_cost_ms=fsum(
            float(costs[i]) for i in selected_indices
        ),
        baseline_name="random_matched_count",
    )

def random_matched_budget(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
    budget_ms: float,
    seed: int,
) -> BaselineResult:
    """Randomized greedy packing with a strict cost cap.

    Prototype only — not M2-frozen.
    """
    ids, costs = _validate_candidates(test_ids, costs_ms)

    if (
        isinstance(budget_ms, (bool, np.bool_))
        or not isinstance(
            budget_ms, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(budget_ms)
        or budget_ms <= 0
    ):
        raise ValueError("Invalid budget_ms.")

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("Invalid random seed.")

    budget = float(budget_ms)

    # Stable canonical order before randomization.
    canonical = sorted(
        range(len(ids)),
        key=lambda i: ids[i],
    )

    rng = np.random.default_rng(int(seed))
    positions = rng.permutation(len(ids))

    selected = []
    spent = 0.0

    for position in positions:
        i = canonical[int(position)]
        cost = float(costs[i])

        # Check affordability before adding, avoiding overflow.
        if cost <= budget - spent:
            selected.append(i)
            spent = fsum((spent, cost))

    selected.sort(key=lambda i: ids[i])

    estimated_cost = fsum(
        float(costs[i]) for i in selected
    )

    if estimated_cost > budget:
        raise RuntimeError("Selected cost exceeds budget.")

    return BaselineResult(
        selected_test_ids=tuple(ids[i] for i in selected),
        estimated_cost_ms=estimated_cost,
        baseline_name="random_matched_budget",
    )

# --- WP4.5 historical selection baselines ---


def _historical_select(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
    failure_rates: Sequence[float],
    k: int,
    use_duration: bool,
) -> BaselineResult:
    """Shared historical baseline implementation.

    PROTOTYPE ONLY — NOT M2-FROZEN.
    """
    ids, costs = _validate_candidates(
        test_ids,
        costs_ms,
    )

    rates = np.asarray(failure_rates, dtype=float)

    if rates.ndim != 1 or len(rates) != len(ids):
        raise ValueError(
            "Failure-rate shape or length mismatch."
        )

    if not np.isfinite(rates).all():
        raise ValueError(
            "Historical failure rates must be finite."
        )

    if ((rates < 0) | (rates > 1)).any():
        raise ValueError(
            "Historical failure rates must be within [0, 1]."
        )

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or not (1 <= k <= len(ids))
    ):
        raise ValueError(
            "k must be an integer within [1, candidate count]."
        )

    if use_duration:
        # Logarithmic ranking avoids division overflow.
        # log(rate / cost) = log(rate) - log(cost).
        # Zero failure rate has the lowest possible rank.
        scores = np.full(len(ids), -np.inf, dtype=float)
        positive = rates > 0

        scores[positive] = (
            np.log(rates[positive])
            - np.log(costs[positive])
        )

        baseline_name = "history_duration"
    else:
        scores = rates
        baseline_name = "failure_history"

    # Descending historical score; ascending ID on ties.
    order = sorted(
        range(len(ids)),
        key=lambda i: (
            -float(scores[i]),
            ids[i],
        ),
    )

    selected = order[:int(k)]

    return BaselineResult(
        selected_test_ids=tuple(
            ids[i] for i in selected
        ),
        estimated_cost_ms=fsum(
            float(costs[i]) for i in selected
        ),
        baseline_name=baseline_name,
    )


def failure_history_select(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
    failure_rates: Sequence[float],
    k: int,
) -> BaselineResult:
    """Select k tests with highest historical failure rates."""
    return _historical_select(
        test_ids=test_ids,
        costs_ms=costs_ms,
        failure_rates=failure_rates,
        k=k,
        use_duration=False,
    )


def history_duration_select(
    test_ids: Sequence[str],
    costs_ms: Sequence[float],
    failure_rates: Sequence[float],
    k: int,
) -> BaselineResult:
    """Select k tests by historical failure rate per unit cost."""
    return _historical_select(
        test_ids=test_ids,
        costs_ms=costs_ms,
        failure_rates=failure_rates,
        k=k,
        use_duration=True,
    )