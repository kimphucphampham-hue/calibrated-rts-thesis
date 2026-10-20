"""Historical cost preparation for RQ2.

PROTOTYPE ONLY — cost policy is NOT M2-frozen.

The adapter handles one project/build per call.
Fallback uses positive realized durations from strictly
earlier builds of the same project.

The current target build's realized duration is never
used to estimate its selection-time cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = {
    "project_id",
    "build_index",
    "test_id",
    "execution_time_mean_history_ms",
}

HISTORY_COLUMNS = {
    "project_id",
    "build_index",
    "execution_time_ms",
}


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns: "
            + ", ".join(sorted(missing))
        )


def estimate_build_costs(
    target: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare positive estimated costs for one target build.

    Preserves valid positive historical per-test estimates.
    Replaces missing/zero estimates using the median of
    positive realized durations from strictly earlier
    builds of the same project.

    The fallback is provisional pending M2 review.
    """
    _require_columns(
        target, TARGET_COLUMNS, "target"
    )
    _require_columns(
        history, HISTORY_COLUMNS, "history"
    )

    if target.empty:
        raise ValueError("Target build must not be empty.")

    if target["project_id"].isna().any():
        raise ValueError("Null target project_id.")

    if target["project_id"].nunique() != 1:
        raise ValueError(
            "Expected exactly one target project."
        )

    if target["test_id"].isna().any():
        raise ValueError("Null target test_id.")

    if target["test_id"].duplicated().any():
        raise ValueError("Duplicate target test_id.")

    target_indices = pd.to_numeric(
        target["build_index"],
        errors="raise",
    )

    if (
        not np.isfinite(target_indices.to_numpy(
            dtype=float
        )).all()
        or target_indices.nunique() != 1
    ):
        raise ValueError(
            "Expected one finite target build_index."
        )

    target_build = target_indices.iloc[0]
    project = target["project_id"].iloc[0]

    estimated = pd.to_numeric(
        target["execution_time_mean_history_ms"],
        errors="raise",
    ).astype(float)

    if np.isinf(estimated.to_numpy()).any():
        raise ValueError(
            "Infinite historical estimated cost."
        )

    if (estimated < 0).any():
        raise ValueError(
            "Negative historical estimated cost."
        )

    needs_fallback = estimated.isna() | estimated.eq(0)

    result = target.copy()

    if not needs_fallback.any():
        result["estimated_cost_ms"] = estimated
        return result

    history_indices = pd.to_numeric(
        history["build_index"],
        errors="raise",
    )

    history_durations = pd.to_numeric(
        history["execution_time_ms"],
        errors="raise",
    ).astype(float)

    if not np.isfinite(
        history_indices.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid history build_index."
        )

    if np.isinf(history_durations.to_numpy()).any():
        raise ValueError(
            "Infinite historical realized duration."
        )

    if (history_durations < 0).any():
        raise ValueError(
            "Negative historical realized duration."
        )

    eligible = (
        history["project_id"].eq(project)
        & history_indices.lt(target_build)
        & history_durations.gt(0)
    )

    eligible_durations = history_durations.loc[
        eligible
    ]

    if eligible_durations.empty:
        raise ValueError(
            "No eligible positive project duration "
            "from strictly earlier builds."
        )

    fallback = float(eligible_durations.median())

    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError(
            "Historical fallback must be positive."
        )

    result["estimated_cost_ms"] = estimated.mask(
        needs_fallback,
        fallback,
    )

    if (
        not np.isfinite(
            result["estimated_cost_ms"].to_numpy(
                dtype=float
            )
        ).all()
        or (result["estimated_cost_ms"] <= 0).any()
    ):
        raise RuntimeError(
            "Adapter produced invalid estimated costs."
        )

    return result
