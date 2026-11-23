"""Pre-execution F1/F2/F3 materialization from an observed historical prefix.

This module REUSES frozen build_history_features.compute_history_features.
It does not accept an evaluation oracle and never returns current-build outcomes.
The target's placeholder verdict/duration are internal technical sentinels only.

Assumption: observed_history contains complete earlier-build outcomes under the
PROPOSED retrospective full-information logical regime (M2-08 PENDING).
Actual CI result-completion availability is NOT established. This component
is pre-freeze implementation evidence, NOT a final longitudinal runner.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.build_history_features import (
    FEATURE_COLUMNS,
    SUPPORT_COLUMNS,
    compute_history_features,
)

IDENTITY: Tuple[str, ...] = ("project_id", "build_id", "build_index", "test_id")
HISTORY_COLUMNS = set(IDENTITY) | {"build_started_at", "failed", "execution_time_ms"}
TARGET_COLUMNS = set(IDENTITY) | {"build_started_at", "changed_entity_count"}
TARGET_F1_ONLY_COLUMNS = set(IDENTITY) | {"build_started_at"}
F1_ONLY_FEATURE_COLUMNS = tuple(FEATURE_COLUMNS[:7])
DECISION_COLUMNS = tuple(dict.fromkeys(
    (*IDENTITY, "build_started_at", *FEATURE_COLUMNS, *SUPPORT_COLUMNS)
))
POST_EXECUTION = frozenset({"failed", "execution_time_ms", "verdict_raw", "duration"})
F1_ONLY_DECISION_COLUMNS = tuple(dict.fromkeys(
    (*IDENTITY, "build_started_at", *F1_ONLY_FEATURE_COLUMNS, *SUPPORT_COLUMNS)
))


def _validate_identity(frame: pd.DataFrame, label: str) -> tuple:
    """Check logical build identities before touching the frozen feature code."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    if frame.empty:
        return None, set()
    for col in ("project_id", "build_id", "test_id"):
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} has invalid {col}")
    indices = pd.to_numeric(frame["build_index"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(indices).all() or (indices < 1).any()
            or (indices != np.floor(indices)).any()):
        raise ValueError(f"{label} has invalid build_index")
    if frame["project_id"].astype(str).nunique() != 1:
        raise ValueError(f"{label} must contain one project")
    if frame.duplicated(list(IDENTITY)).any() or frame.duplicated(
        ["project_id", "build_index", "test_id"]
    ).any():
        raise ValueError(f"{label} has duplicate test identities")
    if (frame.groupby("build_index")["build_id"].nunique().gt(1).any()
            or frame.groupby("build_id")["build_index"].nunique().gt(1).any()):
        raise ValueError(f"{label} has inconsistent build_id/build_index mapping")
    timestamps = pd.to_datetime(frame["build_started_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label} has invalid build_started_at")
    checks = pd.DataFrame({"idx": indices.astype("int64"), "ts": timestamps})
    if checks.groupby("idx")["ts"].nunique().gt(1).any():
        raise ValueError(f"{label} has inconsistent build timestamps")
    ordered_ts = checks.drop_duplicates("idx").sort_values("idx")["ts"]
    if not ordered_ts.is_monotonic_increasing:
        raise ValueError(f"{label} has nonmonotonic build_started_at")
    return str(frame["project_id"].iloc[0]), set(indices.astype(int))


def materialize_target_preexecution_features(
    observed_history: pd.DataFrame,
    target_context: pd.DataFrame,
    *,
    feature_scope: str = "F1_F2_F3",
) -> pd.DataFrame:
    """Produce one target build's frozen features without target oracle inputs.

    observed_history: canonical, fully observed rows from STRICTLY EARLIER
      logical builds (project/build/test/start time/failed/duration). This
      must be a retrospective prefix, not a partial-selection feedback log.
    target_context: in default mode, canonical keys, started-at and
      pre-execution F2 `changed_entity_count`; in explicit F1-only mode,
      canonical keys and started-at ONLY. It MUST NOT contain outcomes,
      realized cost, or precomputed history features.

    `compute_history_features` requires numeric target failed/duration even
    though the frozen formulas use strictly prior observations. Internal zero
    sentinels satisfy that legacy signature and are discarded; this is safe
    only because the target is the last build and there is one row per test.
    """
    if feature_scope not in ("F1_F2_F3", "F1_ONLY_V1"):
        raise ValueError("Unsupported feature_scope for causal materialization")
    f1_only = feature_scope == "F1_ONLY_V1"
    required_target = TARGET_F1_ONLY_COLUMNS if f1_only else TARGET_COLUMNS
    if not isinstance(observed_history, pd.DataFrame):
        raise ValueError("observed_history must be a DataFrame")
    if not isinstance(target_context, pd.DataFrame) or target_context.empty:
        raise ValueError("target_context must be a nonempty DataFrame")
    missing_hist = HISTORY_COLUMNS.difference(observed_history.columns)
    missing_target = required_target.difference(target_context.columns)
    if missing_hist or missing_target:
        raise ValueError(
            f"Missing canonical columns: history={sorted(missing_hist)}, "
            f"target={sorted(missing_target)}"
        )
    unexpected_target = set(target_context.columns).difference(required_target)
    if unexpected_target:
        raise ValueError(f"target_context has unexpected fields: {sorted(unexpected_target)}")

    project, target_indices = _validate_identity(target_context, "target_context")
    if len(target_indices) != 1:
        raise ValueError("target_context must contain exactly one target build")
    target_index = next(iter(target_indices))
    old_project, history_indices = _validate_identity(observed_history, "observed_history")
    if old_project is not None and old_project != project:
        raise ValueError("observed_history and target_context projects differ")
    if history_indices and max(history_indices) >= target_index:
        raise ValueError("History must be strictly earlier than target build_index")

    if not observed_history.empty:
        previous_times = pd.to_datetime(observed_history["build_started_at"], utc=True)
        target_time = pd.to_datetime(target_context["build_started_at"], utc=True).iloc[0]
        if previous_times.max() > target_time:
            raise ValueError("Historical build_started_at follows target start")
        y = pd.to_numeric(observed_history["failed"], errors="raise")
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError("History has invalid failure labels")
        durations = pd.to_numeric(observed_history["execution_time_ms"], errors="raise")
        if (durations.isna().any() or not np.isfinite(durations.to_numpy(dtype=float)).all()
                or (durations < 0).any() or (durations != np.floor(durations)).any()):
            raise ValueError("History has invalid realized durations")

    if not f1_only:
        change_count = pd.to_numeric(target_context["changed_entity_count"], errors="raise")
        defined = change_count.dropna()
        if (not np.isfinite(defined.to_numpy(dtype=float)).all()
                or (defined < 0).any() or (defined != np.floor(defined)).any()):
            raise ValueError("Target has invalid changed_entity_count")

    # Select only the required source fields: do not accidentally reuse
    # previously computed or label-dependent columns from historical frames.
    historical = observed_history.loc[:, list(IDENTITY) + [
        "build_started_at", "failed", "execution_time_ms"
    ]].copy(deep=True)
    if "changed_entity_count" in observed_history.columns:
        historical["changed_entity_count"] = observed_history["changed_entity_count"].copy()
    else:
        historical["changed_entity_count"] = np.nan

    current = target_context.loc[:, list(IDENTITY) + [
        "build_started_at"
    ]].copy(deep=True)
    # Missing F2 is an internal sentinel solely to reuse the historical
    # F1 computation; no F2/F3 fields are returned in F1-only mode.
    current["changed_entity_count"] = (
        np.nan if f1_only else target_context["changed_entity_count"].copy()
    )
    # Private sentinels. No target oracle argument exists in this API.
    current["failed"] = 0
    current["execution_time_ms"] = 0
    joined = pd.concat([historical, current], ignore_index=True)

    materialized = compute_history_features(joined)
    decision_columns = F1_ONLY_DECISION_COLUMNS if f1_only else DECISION_COLUMNS
    subset = materialized.loc[
        materialized["build_index"].eq(target_index)
        & materialized["project_id"].astype(str).eq(project),
        list(decision_columns),
    ].copy(deep=True)
    if len(subset) != len(current):
        raise AssertionError("Frozen feature transform lost target rows")
    if set(subset["test_id"].astype(str)) != set(current["test_id"].astype(str)):
        raise AssertionError("Frozen feature transform changed target test identity")
    if POST_EXECUTION.intersection(subset.columns):
        raise AssertionError("Post-execution oracle field leaked into decision view")
    return subset.sort_values("test_id", kind="mergesort").reset_index(drop=True)
