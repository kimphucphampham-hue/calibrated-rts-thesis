"""Synthetic-only Option A per-build boundary prototype (NOT final evidence).

A probability callback sees strictly earlier full history and a decision view
without target/future failed labels or realized execution durations. The
benchmark oracle is accessed only after selected IDs have been fixed.

This models ordinal full-history availability, NOT real CI completion timing.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback
from src.tcp_ci.rq2_cold_start_reporting import evaluate_post_selection

REQUIRED = (
    "project_id", "build_index", "test_id", "failed", "execution_time_ms"
)
OBSERVATION_REGIME = "retrospective_full_information"


@dataclass(frozen=True)
class PrototypeStep:
    build_index: int
    selected_test_ids: Tuple[str, ...]
    selection_mode: str
    report: Dict


def run_retrospective_prototype(
    records: pd.DataFrame,
    probability_provider: Callable[[pd.DataFrame, pd.DataFrame], object],
    rho: float,
    *,
    start_build_index: int = 1,
) -> Tuple[PrototypeStep, ...]:
    """Run a *synthetic-only* RQ2 decision/scoring bridge.

    Provider(history, decision_view) must return aligned probabilities. It is
    NOT an RQ1 model adapter; replacing it with RQ1 requires a separate WP.
    Input contains canonical test/build outcomes for offline scoring. No
    skipped-test feedback simulator or CI completion timestamp is modeled.
    """
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("Expected nonempty canonical records.")
    if not callable(probability_provider):
        raise TypeError("probability_provider must be callable.")
    if isinstance(start_build_index, bool) or not isinstance(
        start_build_index, (int, np.integer)
    ) or start_build_index < 1:
        raise ValueError("start_build_index must be a positive integer.")
    missing = set(REQUIRED) - set(records.columns)
    if missing:
        raise ValueError("Missing canonical columns: {}".format(sorted(missing)))

    # Construct a minimal internal oracle; ignore ALL other supplied fields.
    data = records.loc[:, list(REQUIRED)].copy(deep=True)
    if data["project_id"].isna().any() or data["test_id"].isna().any():
        raise ValueError("Missing project_id/test_id.")
    data["project_id"] = data["project_id"].astype(str)
    data["test_id"] = data["test_id"].astype(str)
    if (data["test_id"].str.strip() == "").any():
        raise ValueError("Empty test_id.")
    if data["project_id"].nunique() != 1:
        raise ValueError("Exactly one project per prototype run is required.")

    indices = pd.to_numeric(data["build_index"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or (indices < 1).any() or (indices != np.floor(indices)).any():
        raise ValueError("build_index must be a positive finite integer.")
    data["build_index"] = indices.astype(np.int64)
    if data.duplicated(["project_id", "build_index", "test_id"]).any():
        raise ValueError("Duplicate canonical (project, build, test) identity.")
    outcomes = pd.to_numeric(data["failed"], errors="raise")
    if not outcomes.isin([0, 1]).all():
        raise ValueError("failed must be binary.")
    data["failed"] = outcomes.astype(int)
    durations = pd.to_numeric(data["execution_time_ms"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("execution_time_ms must be finite and nonnegative.")
    data["execution_time_ms"] = durations
    data = data.sort_values(["build_index", "test_id"], kind="mergesort")

    steps = []
    for b in sorted(data["build_index"].unique()):
        b = int(b)
        if b < start_build_index:
            continue
        # Strictly earlier full-history observations: Option A ASSUMPTION.
        history = data.loc[data["build_index"] < b].copy(deep=True)
        target_oracle = data.loc[data["build_index"] == b].copy(deep=True)
        if target_oracle.empty:
            raise AssertionError("Internal target build is empty.")

        # Generate decision-time cost from earlier same-test execution history.
        # Full RQ1 feature construction is deliberately outside this prototype.
        means = history.groupby("test_id")["execution_time_ms"].mean()
        decision_view = target_oracle.loc[:, [
            "project_id", "build_index", "test_id"
        ]].copy(deep=True)
        decision_view["execution_time_mean_history_ms"] = (
            decision_view["test_id"].map(means).astype(float)
        )

        # The callback never receives target/future outcomes or durations.
        probabilities = np.asarray(
            probability_provider(history.copy(deep=True), decision_view.copy(deep=True)),
            dtype=float,
        )
        if probabilities.ndim != 1 or len(probabilities) != len(decision_view):
            raise ValueError("Expected one probability per target candidate.")
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise ValueError("Probabilities must be finite and between 0 and 1.")

        selection = select_with_cold_start_fallback(
            target=decision_view,
            history=history,
            probabilities=probabilities,
            rho=rho,
        )
        # ONLY NOW does the offline benchmark oracle reach the evaluator.
        report = evaluate_post_selection(
            target=target_oracle,
            selection=selection,
            rho=rho,
            model="synthetic_stub",
            probability="synthetic_stub",
            cost_policy="historical_median",
            baseline="risk_density",
            observation_regime=OBSERVATION_REGIME,
        )
        steps.append(
            PrototypeStep(
                build_index=b,
                selected_test_ids=tuple(selection.selected_test_ids),
                selection_mode=str(selection.selection_mode),
                report=report,
            )
        )
    return tuple(steps)
