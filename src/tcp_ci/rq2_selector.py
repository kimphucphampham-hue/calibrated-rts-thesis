"""RQ2 risk-bounded test selection — implementation prototype.

This module receives prevalidated, strictly positive estimated
costs and decision-time probabilities.

It does not perform historical cost imputation or use outcomes
from the target build.

M2 scientific decisions remain subject to method review.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class SelectionResult:
    selected_test_ids: tuple[str, ...]
    total_risk: float
    retained_risk: float
    estimated_cost_ms: float
    solver_status: str


def _validate_inputs(
    test_ids: Sequence[str],
    probabilities: Sequence[float],
    costs_ms: Sequence[float],
    rho: float,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    ids = list(test_ids)

    if not ids:
        raise ValueError("Test set must not be empty.")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate test IDs.")

    p = np.asarray(probabilities, dtype=float)
    c = np.asarray(costs_ms, dtype=float)

    if p.ndim != 1 or c.ndim != 1:
        raise ValueError("Probabilities and costs must be 1D.")

    if len(ids) != len(p) or len(ids) != len(c):
        raise ValueError("Input lengths do not match.")

    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be within [0, 1].")

    if not np.isfinite(c).all():
        raise ValueError("Estimated costs must be finite.")

    if (c <= 0).any():
        raise ValueError("Estimated costs must be positive.")

    if not np.isfinite(rho) or not (0 < rho <= 1):
        raise ValueError("rho must satisfy 0 < rho <= 1.")

    total_risk = fsum(float(value) for value in p)

    if total_risk <= 0:
        raise ValueError(
            "Zero-total-risk policy is not frozen. "
            "Failing closed until method review."
        )

    return ids, p, c, total_risk


def _make_result(
    ids: list[str],
    p: np.ndarray,
    c: np.ndarray,
    selected_indices: list[int],
    total_risk: float,
    rho: float,
    status: str,
) -> SelectionResult:
    retained_risk = fsum(
        float(p[i]) for i in selected_indices
    )
    estimated_cost = fsum(
        float(c[i]) for i in selected_indices
    )

    required_risk = rho * total_risk

    if retained_risk + 1e-12 * total_risk < required_risk:
        raise RuntimeError(
            "Selection violates risk-retention constraint."
        )

    return SelectionResult(
        selected_test_ids=tuple(
            ids[i] for i in selected_indices
        ),
        total_risk=total_risk,
        retained_risk=retained_risk,
        estimated_cost_ms=estimated_cost,
        solver_status=status,
    )


def risk_density_select(
    test_ids: Sequence[str],
    probabilities: Sequence[float],
    costs_ms: Sequence[float],
    rho: float,
) -> SelectionResult:
    """Greedy risk-density selection; not globally optimal."""
    ids, p, c, total_risk = _validate_inputs(
        test_ids, probabilities, costs_ms, rho
    )

    target = rho * total_risk

    # Positive costs permit division without epsilon.
    # Prototype tie-breaking:
    # 1. Risk density descending.
    # 2. Test ID ascending for exact density ties.
    #
    # This policy is NOT M2-frozen.
    order = sorted(
        range(len(ids)),
        key=lambda i: (
            -float(p[i] / c[i]),
            ids[i],
        ),
    )

    selected_indices: list[int] = []
    retained = 0.0
    compensation = 0.0

    for index in order:
        i = int(index)
        selected_indices.append(i)

        # Kahan compensated summation: O(1) per test.
        increment = float(p[i]) - compensation
        updated = retained + increment
        compensation = (updated - retained) - increment
        retained = updated

        if retained >= target:
            break

    return _make_result(
        ids, p, c, selected_indices,
        total_risk, rho, "heuristic"
    )


def milp_select(
    test_ids: Sequence[str],
    probabilities: Sequence[float],
    costs_ms: Sequence[float],
    rho: float,
) -> SelectionResult:
    """Exact 0-1 formulation; requires solver-proven optimality."""
    ids, p, c, total_risk = _validate_inputs(
        test_ids, probabilities, costs_ms, rho
    )

    n = len(ids)

    # Normalize predicted risk to improve numerical
    # conditioning when probabilities are very small.
    # This is equivalent to the original constraint
    # because total_risk is strictly positive.
    risk_share = p / total_risk

    # Minimize c @ x subject to
    # risk_share @ x >= rho, x_i in {0, 1}.
    result = milp(
        c=c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(
            np.zeros(n),
            np.ones(n),
        ),
        constraints=LinearConstraint(
            risk_share.reshape(1, -1),
            lb=np.array([rho]),
            ub=np.array([np.inf]),
        ),
        options={"disp": False},
    )

    if (
        result.status != 0
        or result.x is None
        or not result.success
    ):
        raise RuntimeError(
            "MILP did not prove optimality: "
            f"status={result.status}; "
            f"message={result.message}"
        )

    if not np.isfinite(result.x).all():
        raise RuntimeError("MILP returned nonfinite solution.")

    if np.max(
        np.abs(result.x - np.rint(result.x))
    ) > 1e-7:
        raise RuntimeError(
            "MILP returned a non-integral solution."
        )

    selected_indices = [
        i for i, value in enumerate(result.x)
        if value >= 0.5
    ]

    return _make_result(
        ids, p, c, selected_indices,
        total_risk, rho, "optimal"
    )
