"""Synthetic-only multi-build RQ1/RQ2 walk-forward integration.

Runs one model/probability/rho condition across consecutive logical decisions.
Each step fits the existing RQ1 helper on preceding observed builds, selects
using existing RQ2 code, scores with an isolated oracle, then advances full
history under a HYPOTHETICAL retrospective full-information convention.

Upstream F1/F2/F3 feature provenance, real CI finish-time availability,
static/periodic RQ3 cadence, and formal method approval are NOT established.
Never use this module for confirmatory CORE results.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq1_rq2_single_build_bridge import (
    SingleBuildResult,
    evaluate_one_build_with_rq1,
)
from src.tcp_ci.build_history_features import compute_history_features
from src.tcp_ci.rq2_causal_feature_materializer import (
    TARGET_COLUMNS,
    materialize_target_preexecution_features,
)

IDENTITY = ("project_id", "build_id", "build_index", "test_id")
ORACLE_FIELDS = IDENTITY + ("failed", "execution_time_ms")
POST_EXECUTION = frozenset({"failed", "execution_time_ms", "verdict_raw", "duration"})
REGIME = "retrospective_full_information"


@dataclass(frozen=True)
class MultiBuildPrototypeResult:
    steps: Tuple[SingleBuildResult, ...]
    development_rows_before: Tuple[int, ...]
    history_rows_before: Tuple[int, ...]
    observation_regime: str = REGIME
    evidence_scope: str = "synthetic_multibuild_integration_prototype"


def _check_frame(frame: pd.DataFrame, name: str, required: set) -> tuple:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    for field in ("project_id", "build_id", "test_id"):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has invalid {field}.")
    numeric = pd.to_numeric(frame["build_index"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (numeric < 1).any()
            or (numeric != np.floor(numeric)).any()):
        raise ValueError(f"{name} has invalid build_index.")
    if frame["project_id"].astype(str).nunique() != 1:
        raise ValueError(f"{name} must contain exactly one project.")
    if frame.duplicated(["project_id", "build_index", "test_id"]).any():
        raise ValueError(f"{name} contains duplicate build/test identities.")
    if frame.groupby("build_index")["build_id"].nunique().gt(1).any():
        raise ValueError(f"{name} has multiple build IDs for a build index.")
    if frame.groupby("build_id")["build_index"].nunique().gt(1).any():
        raise ValueError(f"{name} has multiple build indices for a build ID.")
    return str(frame["project_id"].astype(str).iloc[0]), set(numeric.astype(int))


def run_synthetic_multibuild_rq1_rq2(
    initial_development: pd.DataFrame,
    target_preexecution: pd.DataFrame,
    target_oracle: pd.DataFrame,
    *,
    model_name: str,
    probability_variant: str,
    rho: float,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
    preexecution_mode: str = "supplied_synthetic",
) -> MultiBuildPrototypeResult:
    """Run a bounded, in-memory synthetic chronological full-history prototype.

    The initial development set has historical full test results. The supplied
    target features MUST have been produced causally upstream (not verified by
    this function). The target oracle's values enter development/history only
    *after* that build's selection/scoring, to exercise sequential feedback.
    Re-fitting per build here is an ENGINEERING TEST, not frozen RQ3 cadence.

    Modes:
      supplied_synthetic: legacy synthetic precomputed feature inputs.
      causal_from_history: target contains ONLY canonical pre-execution context;
      re-derive development/target F1/F2/F3 with the frozen historical
      transform using the observed full-information prefix before EACH build.
      M2-08 remains pending; this is not a real CI completion-time simulator.
    """
    if preexecution_mode not in {"supplied_synthetic", "causal_from_history"}:
        raise ValueError("Unsupported preexecution_mode.")
    causal = preexecution_mode == "causal_from_history"
    dev_required = set(ORACLE_FIELDS)
    pre_required = (set(TARGET_COLUMNS) if causal else
                    set(IDENTITY) | {"execution_time_mean_history_ms"})
    if causal:
        required_raw = set(ORACLE_FIELDS) | {"build_started_at", "changed_entity_count"}
        if not isinstance(initial_development, pd.DataFrame) or set(initial_development.columns) != required_raw:
            raise ValueError("Causal development must contain exactly raw canonical and F2 fields.")
        if not isinstance(target_preexecution, pd.DataFrame) or set(target_preexecution.columns) != set(TARGET_COLUMNS):
            raise ValueError("Causal target must contain exactly pre-execution context fields.")
    oracle_required = set(ORACLE_FIELDS)
    seed_project, seed_indices = _check_frame(
        initial_development, "initial_development", dev_required
    )
    pre_project, target_indices = _check_frame(
        target_preexecution, "target_preexecution", pre_required
    )
    oracle_project, oracle_indices = _check_frame(
        target_oracle, "target_oracle", oracle_required
    )
    if len({seed_project, pre_project, oracle_project}) != 1:
        raise ValueError("Development, preexecution, and oracle projects differ.")
    forbidden = POST_EXECUTION.intersection(target_preexecution.columns)
    if forbidden:
        raise ValueError(f"Preexecution contains post-execution fields: {sorted(forbidden)}")
    if set(target_oracle.columns) != oracle_required:
        raise ValueError("Oracle must contain only identity, failed and execution_time_ms.")
    if max(seed_indices) >= min(target_indices):
        raise ValueError("Initial development overlaps target or future builds.")
    if target_indices != oracle_indices:
        raise ValueError("Oracle and preexecution build-index coverage differs.")

    expected_keys = target_preexecution.loc[:, list(IDENTITY)]
    actual_keys = target_oracle.loc[:, list(IDENTITY)]
    key_merge = expected_keys.merge(
        actual_keys, on=list(IDENTITY), how="outer", sort=False,
        validate="one_to_one", indicator=True,
    )
    if len(key_merge) != len(expected_keys) or not key_merge["_merge"].eq("both").all():
        raise ValueError("Oracle and preexecution test/build identities differ.")

    def _binary_labels(frame: pd.DataFrame, name: str) -> None:
        y = pd.to_numeric(frame["failed"], errors="raise")
        if y.isna().any() or not y.isin((0, 1)).all():
            raise ValueError(f"{name} has invalid failure labels.")
        costs = pd.to_numeric(frame["execution_time_ms"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError(f"{name} has invalid realized durations.")

    _binary_labels(initial_development, "initial_development")
    _binary_labels(target_oracle, "target_oracle")

    development = initial_development.copy(deep=True)
    # The sole source of cost history is a strictly earlier, local state.
    steps = []
    development_counts = []
    history_counts = []
    for build_index in sorted(target_indices):
        current_context = target_preexecution.loc[
            pd.to_numeric(target_preexecution["build_index"]) == build_index
        ].copy(deep=True)
        current_context = current_context.sort_values(
            "test_id", kind="mergesort").reset_index(drop=True)
        if causal:
            # Never trust precomputed historical features from the caller.
            # Current target oracle is intentionally unavailable to both calls.
            current_pre = materialize_target_preexecution_features(
                observed_history=development, target_context=current_context,
            )
            development_features = compute_history_features(development.copy(deep=True))
        else:
            current_pre = current_context.copy(deep=True)
            development_features = development.copy(deep=True)
        current_oracle = target_oracle.loc[
            pd.to_numeric(target_oracle["build_index"]) == build_index,
            list(ORACLE_FIELDS),
        ].copy(deep=True)
        history = development.loc[:, [
            "project_id", "build_index", "test_id", "execution_time_ms"
        ]].copy(deep=True)
        if pd.to_numeric(history["build_index"]).ge(build_index).any():
            raise AssertionError("Internal history advanced before target decision.")
        development_counts.append(len(development))
        history_counts.append(len(history))

        # Existing RQ1 adapter masks target truth; existing RQ2 selector uses
        # only current pre-execution inputs plus the earlier history above.
        step = evaluate_one_build_with_rq1(
            development=development_features,
            target_preexecution=current_pre.copy(deep=True),
            history=history,
            target_oracle=current_oracle.copy(deep=True),
            model_name=model_name,
            probability_variant=probability_variant,
            rho=rho,
            rq1_config=rq1_config,
            model_config=model_config,
            calibration_config=calibration_config,
            core_features_config=core_features_config,
        )
        steps.append(step)

        # Feedback is appended ONLY AFTER selection & retrospective scoring.
        # Join by canonical identity, never align oracle outcomes by row order.
        # Advance with RAW pre-execution metadata + oracle, not computed
        # target features (which must be rematerialized from the next prefix).
        realized = current_context.merge(
            current_oracle, on=list(IDENTITY), how="left", sort=False,
            validate="one_to_one", indicator=True,
        )
        if not realized["_merge"].eq("both").all():
            raise AssertionError("Internal oracle identity mismatch.")
        realized = realized.drop(columns="_merge")
        development = pd.concat([development, realized], ignore_index=True)

    return MultiBuildPrototypeResult(
        steps=tuple(steps),
        development_rows_before=tuple(development_counts),
        history_rows_before=tuple(history_counts),
    )
