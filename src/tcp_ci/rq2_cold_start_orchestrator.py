"""Candidate M2-06 cold-start orchestration.

PROTOTYPE ONLY — NOT M2-FROZEN.

Normal path:
    Historical median cost adapter -> RiskDensitySort.

Candidate fallback:
    If the cost adapter cannot find any eligible positive
    earlier-build project duration when a median is needed,
    select all candidate tests.

Other errors are never converted into Retest All.

This module does not use target-build failure labels or
realized execution durations to make a selection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq2_cost_adapter import estimate_build_costs
from src.tcp_ci.rq2_selector import risk_density_select


NO_POSITIVE_HISTORY_ERROR = (
    "No eligible positive project duration "
    "from strictly earlier builds."
)

COLD_START_REASON = "no_positive_prior_cost_history"


@dataclass(frozen=True)
class ColdStartSelectionResult:
    project_id: str
    build_index: str
    selected_test_ids: Tuple[str, ...]
    selection_mode: str
    fallback_reason: Optional[str]
    n_candidates: int
    estimated_selected_cost_ms: Optional[float]
    retained_risk_ratio: float

    def diagnostic_record(self):
        """Return a record suitable for later evaluation reporting."""
        return {
            "project_id": self.project_id,
            "build_index": self.build_index,
            "selection_mode": self.selection_mode,
            "fallback_reason": self.fallback_reason,
            "n_candidates": self.n_candidates,
            "n_selected": len(self.selected_test_ids),
            "estimated_selected_cost_ms": (
                self.estimated_selected_cost_ms
            ),
            "retained_risk_ratio": self.retained_risk_ratio,
        }


def select_with_cold_start_fallback(
    target: pd.DataFrame,
    history: pd.DataFrame,
    probabilities,
    rho: float,
) -> ColdStartSelectionResult:
    """Run a candidate decision-time selection policy.

    Probabilities must be in the same row order as target.

    The fallback is allowed only for the exact
    no-positive-earlier-history cost-adapter error.
    """

    if not isinstance(target, pd.DataFrame):
        raise TypeError("target must be a DataFrame.")

    if target.empty:
        raise ValueError("Target build must not be empty.")

    if "test_id" not in target.columns:
        raise ValueError("Missing target test_id.")

    if target["test_id"].isna().any():
        raise ValueError("Null target test_id.")

    test_ids = tuple(
        str(value) for value in target["test_id"].tolist()
    )

    if len(set(test_ids)) != len(test_ids):
        raise ValueError(
            "Duplicate test_id after string normalization."
        )

    if isinstance(probabilities, pd.Series):
        if not probabilities.index.equals(target.index):
            raise ValueError(
                "Probabilities and target index mismatch."
            )

    try:
        p = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid probability values."
        ) from exc

    if p.ndim != 1 or len(p) != len(target):
        raise ValueError(
            "Probabilities must match target row count."
        )

    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Probabilities must be finite and within [0, 1]."
        )

    if (
        isinstance(rho, (bool, np.bool_))
        or not isinstance(
            rho,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(rho)
        or rho <= 0
        or rho > 1
    ):
        raise ValueError(
            "rho must be finite and within (0, 1]."
        )

    total_risk = float(np.sum(p))

    # M2-02 is a separate pending decision.
    # Do not silently treat zero predicted risk as cold start.
    if total_risk <= 0:
        raise ValueError(
            "Zero total predicted risk: M2-02 is pending."
        )

    try:
        costs = estimate_build_costs(
            target=target,
            history=history,
            zero_cost_floor_ms=None,
        )

    except ValueError as exc:
        # Intentionally exact matching.
        # A generic except ValueError -> Retest All
        # would hide unrelated data and implementation errors.
        if str(exc) != NO_POSITIVE_HISTORY_ERROR:
            raise

        return ColdStartSelectionResult(
            project_id=str(target["project_id"].iloc[0]),
            build_index=str(target["build_index"].iloc[0]),
            selected_test_ids=test_ids,
            selection_mode="fallback_retest_all",
            fallback_reason=COLD_START_REASON,
            n_candidates=len(test_ids),
            estimated_selected_cost_ms=None,
            retained_risk_ratio=1.0,
        )

    if not costs.index.equals(target.index):
        raise RuntimeError(
            "Cost adapter changed target index alignment."
        )

    cost_ids = tuple(
        str(value) for value in costs["test_id"].tolist()
    )

    if cost_ids != test_ids:
        raise RuntimeError(
            "Cost adapter changed target test alignment."
        )

    estimated_costs = costs[
        "estimated_cost_ms"
    ].to_numpy(dtype=float)

    selection = risk_density_select(
        test_ids=test_ids,
        probabilities=p,
        costs_ms=estimated_costs,
        rho=float(rho),
    )

    selected_ids = tuple(
        str(value)
        for value in selection.selected_test_ids
    )

    if (
        len(selected_ids) != len(set(selected_ids))
        or not set(selected_ids).issubset(set(test_ids))
    ):
        raise RuntimeError(
            "Selector returned invalid test identities."
        )

    selected_set = set(selected_ids)

    selected_mask = np.asarray(
        [test_id in selected_set for test_id in test_ids],
        dtype=bool,
    )

    retained_ratio = float(
        np.sum(p[selected_mask]) / total_risk
    )

    if retained_ratio + 1e-9 < rho:
        raise RuntimeError(
            "Selector violated predicted risk coverage."
        )

    return ColdStartSelectionResult(
        project_id=str(target["project_id"].iloc[0]),
        build_index=str(target["build_index"].iloc[0]),
        selected_test_ids=selected_ids,
        selection_mode="normal_risk_density",
        fallback_reason=None,
        n_candidates=len(test_ids),
        estimated_selected_cost_ms=float(
            np.sum(estimated_costs[selected_mask])
        ),
        retained_risk_ratio=retained_ratio,
    )
