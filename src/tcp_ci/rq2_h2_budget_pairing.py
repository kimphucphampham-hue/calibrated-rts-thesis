"""Pre-freeze H2 comparable-estimated-budget comparators (offline, no oracle).

This module composes existing risk-baseline and seed helpers without modifying
those algorithms. A shared *estimated* cost cap is not an exact-spend match.
Historical failure rates and positive costs must be causally materialized by
the caller; no target-build outcome or realized duration is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence, Tuple

from src.tcp_ci.rq2_baseline_pairing import (
    RandomPairedBuild,
    WORKING_BASE_SEED,
    WORKING_REPETITIONS,
    pair_random_baselines_for_build,
)
from src.tcp_ci.rq2_baselines import (
    BaselineResult,
    failure_history_select,
    history_duration_select,
)


@dataclass(frozen=True)
class H2BudgetPairedBuild:
    """All methods share one reference selector's estimated-cost cap."""

    random_pair: RandomPairedBuild
    failure_history_budget: BaselineResult
    history_duration_budget: BaselineResult
    random_budget_utilizations: Tuple[float, ...]
    failure_history_budget_utilization: float
    history_duration_budget_utilization: float
    h2_primary_rho: bool


def _pack_ranked_ids(
    ranked_test_ids: Sequence[str],
    positive_costs_by_id: dict[str, float],
    budget_ms: float,
    baseline_name: str,
) -> BaselineResult:
    """Reuse historical rank order, skipping items over remaining budget."""
    chosen: list[str] = []
    running_cost = 0.0
    for test_id in ranked_test_ids:
        cost = positive_costs_by_id[test_id]
        if cost <= budget_ms - running_cost:
            chosen.append(test_id)
            running_cost = fsum((running_cost, cost))
    total_cost = fsum(positive_costs_by_id[t] for t in chosen)
    if total_cost > budget_ms:
        raise RuntimeError("Historical comparator exceeded reference budget")
    return BaselineResult(
        selected_test_ids=tuple(chosen),
        estimated_cost_ms=total_cost,
        baseline_name=baseline_name,
    )


def pair_h2_budget_comparators_for_build(
    *,
    project_id: str,
    build_index: int,
    condition_key: Tuple[str, ...],
    test_ids: Sequence[str],
    estimated_costs_ms: Sequence[float],
    historical_failure_rates: Sequence[float],
    reference_selected_test_ids: Sequence[str],
    base_seed: int = WORKING_BASE_SEED,
) -> H2BudgetPairedBuild:
    """Pair cost-capped Random (30x) and both historical rankings.

    Uses existing baseline score order and M2-05 seed-v1; does not inspect
    target labels, target realized costs, or actual CI result completion times.
    Missing/unavailable cost is an upstream cold-start fallback, not imputed
    here. History rates must be finite and reflect *only* admitted history.
    """
    if not isinstance(condition_key, tuple) or len(condition_key) != 5:
        raise ValueError("Expected canonical five-field condition_key tuple")
    if condition_key[2] not in ("rho=0.90", "rho=0.95", "rho=0.99"):
        raise ValueError("rho condition must be canonical 0.90, 0.95 or 0.99")

    # Existing pairing is the source of truth for candidate/cost validation,
    # reference selected-cost cap, deterministic seeds and 30 repeats.
    paired = pair_random_baselines_for_build(
        project_id=project_id,
        build_index=build_index,
        condition_key=condition_key,
        test_ids=test_ids,
        estimated_costs_ms=estimated_costs_ms,
        reference_selected_test_ids=reference_selected_test_ids,
        include_budget_sensitivity=True,
        base_seed=base_seed,
    )
    ids = tuple(test_ids)
    costs = tuple(float(v) for v in estimated_costs_ms)
    budget = paired.reference_estimated_cost_ms

    # Both existing historical selectors validate finite [0,1] rates and
    # preserve their own deterministic ranking / test-id tie break.
    failure_ranked = failure_history_select(
        ids, costs, historical_failure_rates, k=len(ids)
    ).selected_test_ids
    duration_ranked = history_duration_select(
        ids, costs, historical_failure_rates, k=len(ids)
    ).selected_test_ids
    lookup = dict(zip(ids, costs))
    failure_result = _pack_ranked_ids(
        failure_ranked, lookup, budget, "failure_history_matched_budget"
    )
    duration_result = _pack_ranked_ids(
        duration_ranked, lookup, budget, "history_duration_matched_budget"
    )

    if len(paired.random_replicates) != WORKING_REPETITIONS:
        raise RuntimeError("Unexpected number of seeded random repetitions")
    random_utilizations = []
    for replicate in paired.random_replicates:
        result = replicate.budget_capped
        if result is None or result.estimated_cost_ms > budget:
            raise RuntimeError("Budget-capped Random result missing or overspent")
        random_utilizations.append(result.estimated_cost_ms / budget)

    return H2BudgetPairedBuild(
        random_pair=paired,
        failure_history_budget=failure_result,
        history_duration_budget=duration_result,
        random_budget_utilizations=tuple(random_utilizations),
        failure_history_budget_utilization=(
            failure_result.estimated_cost_ms / budget
        ),
        history_duration_budget_utilization=(
            duration_result.estimated_cost_ms / budget
        ),
        h2_primary_rho=condition_key[2] in ("rho=0.95", "rho=0.99"),
    )
