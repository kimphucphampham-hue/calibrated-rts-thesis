"""Post-selection reporting for candidate M2-06 orchestration.

PROTOTYPE ONLY — NOT FINAL RQ2 EVIDENCE.

Failure labels and realized durations may be used here only
AFTER selection. They must never influence selection-time costs.
"""

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

from src.tcp_ci.rq2_cold_start_orchestrator import (
    COLD_START_REASON,
    ColdStartSelectionResult,
)


NORMAL_MODE = "normal_risk_density"
FALLBACK_MODE = "fallback_retest_all"

EVIDENCE_SCOPE = "m2_06_reporting_prototype"

LEGACY_GROUP_KEYS = ["project", "model", "probability", "rho"]
CONDITION_KEYS = ["cost_policy", "baseline", "observation_regime"]
GROUP_KEYS = LEGACY_GROUP_KEYS + CONDITION_KEYS
SUPPORTED_OBSERVATION_REGIME = "retrospective_full_information"


def _canonical_build_index(value):
    """Use an exact positive integer for duplicate-build identity."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Invalid report build_index.")
    try:
        number = Decimal(str(value).strip())
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("Invalid report build_index.") from exc
    if (not number.is_finite() or number <= 0
            or number != number.to_integral_value()):
        raise ValueError("Invalid report build_index.")
    return int(number)


def _single_value(frame, column):
    if column not in frame.columns:
        raise ValueError(
            f"Missing evaluation column: {column}"
        )

    values = frame[column]

    if (
        values.isna().any()
        or values.astype(str).nunique() != 1
    ):
        raise ValueError(
            f"Expected one non-null {column}."
        )

    return str(values.iloc[0])


def _numeric_values(frame, column, binary=False):
    if column not in frame.columns:
        raise ValueError(
            f"Missing evaluation column: {column}"
        )

    try:
        values = pd.to_numeric(
            frame[column],
            errors="raise",
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {column}."
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            f"Nonfinite {column}."
        )

    if binary:
        if not np.isin(
            values, [0.0, 1.0]
        ).all():
            raise ValueError(
                "failed must contain only 0 or 1."
            )
    elif (values < 0).any():
        raise ValueError(
            f"Negative {column}."
        )

    return values


def evaluate_post_selection(
    target: pd.DataFrame,
    selection: ColdStartSelectionResult,
    *,
    rho: float,
    model: str = "unspecified",
    probability: str = "unspecified",
    cost_policy=None,
    baseline=None,
    observation_regime=None,
) -> dict:
    """Evaluate selection using recorded full-build outcomes.

    Undefined RTR, Failure Recall, and MFR are represented
    by NaN rather than arbitrary zero/one values.
    """

    if not isinstance(target, pd.DataFrame) or target.empty:
        raise ValueError(
            "Evaluation target must be a nonempty DataFrame."
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

    if not isinstance(model, str) or not model.strip():
        raise ValueError("Invalid model name.")

    if (
        not isinstance(probability, str)
        or not probability.strip()
    ):
        raise ValueError("Invalid probability name.")

    # Identifiers only: these labels do not establish any M2 policy.
    condition = dict(cost_policy=cost_policy, baseline=baseline,
                     observation_regime=observation_regime)
    provided = [value is not None for value in condition.values()]
    if any(provided) and not all(provided):
        raise ValueError("Incomplete experimental condition metadata.")
    if all(provided):
        if any(not isinstance(v, str) or not v.strip()
               for v in condition.values()):
            raise ValueError("Invalid experimental condition metadata.")
        condition = {k: v.strip() for k, v in condition.items()}
        if condition["observation_regime"] != SUPPORTED_OBSERVATION_REGIME:
            raise ValueError(
                "Unsupported observation regime for full-information reporting."
            )
    else:
        condition = {}  # Preserve unlabeled WP5.11 reporting behavior.

    project = _single_value(target, "project_id")
    build_index = _single_value(target, "build_index")

    build_id = (
        _single_value(target, "build_id")
        if "build_id" in target.columns
        else None
    )

    if (
        project != selection.project_id
        or build_index != selection.build_index
    ):
        raise ValueError(
            "Selection and evaluation build identity mismatch."
        )

    if (
        "test_id" not in target.columns
        or target["test_id"].isna().any()
    ):
        raise ValueError(
            "Missing or null evaluation test_id."
        )

    test_ids = tuple(
        target["test_id"].astype(str).tolist()
    )

    if len(test_ids) != len(set(test_ids)):
        raise ValueError(
            "Duplicate evaluation test_id."
        )

    if any("|" in value for value in test_ids):
        raise ValueError(
            "test_id contains reserved selection delimiter."
        )

    selected_ids = tuple(
        str(value)
        for value in selection.selected_test_ids
    )

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            "Duplicate selected test IDs."
        )

    selected_set = set(selected_ids)
    available_set = set(test_ids)

    if not selected_set.issubset(available_set):
        raise ValueError(
            "Selected test IDs outside evaluation target."
        )

    if selection.n_candidates != len(test_ids):
        raise ValueError(
            "Selection candidate count mismatch."
        )

    mode = selection.selection_mode

    if mode not in (NORMAL_MODE, FALLBACK_MODE):
        raise ValueError(
            "Unknown selection mode."
        )

    is_fallback = mode == FALLBACK_MODE
    estimated = selection.estimated_selected_cost_ms

    if is_fallback:
        if (
            selected_set != available_set
            or selection.fallback_reason != COLD_START_REASON
            or estimated is not None
        ):
            raise ValueError(
                "Invalid fallback selection diagnostic."
            )

    else:
        if selection.fallback_reason is not None:
            raise ValueError(
                "Unexpected normal-selection fallback reason."
            )

        if estimated is None:
            raise ValueError(
                "Normal selection requires estimated cost."
            )

        try:
            estimated = float(estimated)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid normal estimated cost."
            ) from exc

        if not np.isfinite(estimated) or estimated <= 0:
            raise ValueError(
                "Invalid normal estimated cost."
            )

    coverage = float(selection.retained_risk_ratio)

    if (
        not np.isfinite(coverage)
        or coverage < 0
        or coverage > 1 + 1e-9
    ):
        raise ValueError(
            "Invalid retained risk ratio."
        )

    if is_fallback:
        if not np.isclose(
            coverage, 1.0, rtol=0, atol=1e-9
        ):
            raise ValueError(
                "Fallback must retain all predicted risk."
            )
    elif coverage + 1e-9 < rho:
        raise ValueError(
            "Normal selection violated risk coverage."
        )

    # Ground truth is used only after selection.
    failures = _numeric_values(
        target, "failed", binary=True
    )

    durations = _numeric_values(
        target, "execution_time_ms"
    )

    selected_mask = np.asarray(
        [
            test_id in selected_set
            for test_id in test_ids
        ],
        dtype=bool,
    )

    n_tests = len(test_ids)
    n_selected = int(selected_mask.sum())

    total_failures = int(failures.sum())

    captured_failures = int(
        failures[selected_mask].sum()
    )

    missed_failures = (
        total_failures - captured_failures
    )

    full_runtime = float(durations.sum())

    selected_runtime = float(
        durations[selected_mask].sum()
    )

    return {
        "evidence_scope": EVIDENCE_SCOPE,
        "project": project,
        "build_index": build_index,
        "build_id": build_id,
        "model": model,
        "probability": probability,
        "rho": float(rho),
        "selection_mode": mode,
        "fallback_reason": selection.fallback_reason,
        "is_fallback": is_fallback,
        "n_tests": n_tests,
        "n_selected": n_selected,
        "tsr": n_selected / n_tests,
        "estimated_selected_ms": (
            None if is_fallback else estimated
        ),
        "estimated_cost_defined": not is_fallback,
        "observed_selected_ms": selected_runtime,
        "observed_full_ms": full_runtime,
        "rtr": (
            selected_runtime / full_runtime
            if full_runtime > 0
            else np.nan
        ),
        "rtr_defined": full_runtime > 0,
        "total_failures": total_failures,
        "captured_failures": captured_failures,
        "missed_failures": missed_failures,
        "failure_recall": (
            captured_failures / total_failures
            if total_failures > 0
            else np.nan
        ),
        "missed_failure_rate": (
            missed_failures / total_failures
            if total_failures > 0
            else np.nan
        ),
        "failure_metrics_defined": total_failures > 0,
        "retained_risk_ratio": min(coverage, 1.0),
        "selected_test_ids": "|".join(
            sorted(selected_ids)
        ),
        **condition,
    }


def summarize_selection_reports(
    reports, *, require_condition_metadata=False
) -> pd.DataFrame:
    """Aggregate with strict separation of labeled conditions.

    Unlabeled reports are accepted only as legacy prototype inputs.
    Comparative experiments must set require_condition_metadata=True.
    Micro metrics use summed numerators/denominators; fallback is kept.
    """

    frame = pd.DataFrame(reports).copy()
    labeled = any(key in frame.columns for key in CONDITION_KEYS)
    if require_condition_metadata and not labeled:
        raise ValueError("Experimental condition metadata required.")
    if labeled and not set(CONDITION_KEYS).issubset(frame.columns):
        raise ValueError("Incomplete experimental condition metadata.")
    keys = GROUP_KEYS if labeled else LEGACY_GROUP_KEYS

    required = set(
        keys
        + [
            "build_index",
            "selection_mode",
            "fallback_reason",
            "n_tests",
            "n_selected",
            "observed_selected_ms",
            "observed_full_ms",
            "total_failures",
            "captured_failures",
            "missed_failures",
            "rtr_defined",
            "failure_metrics_defined",
        ]
    )

    missing = required - set(frame.columns)

    if frame.empty or missing:
        raise ValueError(
            "Empty or incomplete selection reports: "
            + str(sorted(missing))
        )

    if labeled:
        for field in CONDITION_KEYS:
            valid = frame[field].map(
                lambda v: isinstance(v, str) and bool(v.strip())
            )
            if not valid.all():
                raise ValueError(
                    "Incomplete or invalid experimental condition metadata."
                )
            frame[field] = frame[field].str.strip()

        if not frame["observation_regime"].eq(
            SUPPORTED_OBSERVATION_REGIME
        ).all():
            raise ValueError(
                "Unsupported observation regime for full-information reporting."
            )

    identity = keys + ["build_index"]
    if frame[identity].isna().any().any():
        raise ValueError("Null report grouping/build key.")
    frame["_canonical_build_index"] = frame["build_index"].map(
        _canonical_build_index
    )
    identity = keys + ["_canonical_build_index"]
    if frame.duplicated(subset=identity).any():
        raise ValueError(
            "Duplicate evaluation build within condition."
        )

    if not frame["selection_mode"].isin(
        [NORMAL_MODE, FALLBACK_MODE]
    ).all():
        raise ValueError(
            "Unknown report selection mode."
        )

    if (
        frame["n_tests"].le(0).any()
        or frame["n_selected"].lt(0).any()
        or frame["n_selected"].gt(
            frame["n_tests"]
        ).any()
        or frame["observed_selected_ms"].lt(0).any()
        or frame["observed_full_ms"].lt(0).any()
        or frame["observed_selected_ms"].gt(
            frame["observed_full_ms"] + 1e-6
        ).any()
        or frame["captured_failures"].lt(0).any()
        or frame["missed_failures"].lt(0).any()
        or (
            frame["captured_failures"]
            + frame["missed_failures"]
            != frame["total_failures"]
        ).any()
    ):
        raise ValueError(
            "Invalid selection report accounting."
        )

    if not (
        frame["rtr_defined"].astype(bool)
        == frame["observed_full_ms"].gt(0)
    ).all():
        raise ValueError(
            "Inconsistent runtime metric defined flag."
        )

    if not (
        frame["failure_metrics_defined"].astype(bool)
        == frame["total_failures"].gt(0)
    ).all():
        raise ValueError(
            "Inconsistent failure metric defined flag."
        )

    fallback_rows = frame["selection_mode"].eq(
        FALLBACK_MODE
    )

    if (
        frame.loc[fallback_rows, "n_selected"]
        != frame.loc[fallback_rows, "n_tests"]
    ).any():
        raise ValueError(
            "Fallback report did not select all tests."
        )

    if (
        frame.loc[fallback_rows, "fallback_reason"]
        != COLD_START_REASON
    ).any():
        raise ValueError(
            "Invalid fallback reason."
        )

    summaries = []

    for key, part in frame.groupby(
        keys, sort=True
    ):
        total_tests = int(part["n_tests"].sum())

        selected_tests = int(
            part["n_selected"].sum()
        )

        full_runtime = float(
            part["observed_full_ms"].sum()
        )

        selected_runtime = float(
            part["observed_selected_ms"].sum()
        )

        total_failures = int(
            part["total_failures"].sum()
        )

        captured = int(
            part["captured_failures"].sum()
        )

        missed = int(
            part["missed_failures"].sum()
        )

        n_fallback = int(
            part["selection_mode"]
            .eq(FALLBACK_MODE)
            .sum()
        )

        summaries.append({
            **dict(zip(keys, key)),
            "evidence_scope": EVIDENCE_SCOPE,
            "n_evaluation_builds": len(part),
            "n_fallback_builds": n_fallback,
            "n_normal_builds": (
                len(part) - n_fallback
            ),
            "n_undefined_rtr_builds": int(
                (~part["rtr_defined"]).sum()
            ),
            "n_zero_failure_builds": int(
                (~part["failure_metrics_defined"]).sum()
            ),
            "n_tests": total_tests,
            "n_selected": selected_tests,
            "tsr_micro": (
                selected_tests / total_tests
            ),
            "observed_selected_ms": selected_runtime,
            "observed_full_ms": full_runtime,
            "rtr_micro": (
                selected_runtime / full_runtime
                if full_runtime > 0
                else np.nan
            ),
            "total_failures": total_failures,
            "captured_failures": captured,
            "missed_failures": missed,
            "failure_recall_micro": (
                captured / total_failures
                if total_failures > 0
                else np.nan
            ),
            "missed_failure_rate_micro": (
                missed / total_failures
                if total_failures > 0
                else np.nan
            ),
        })

    return (
        pd.DataFrame(summaries)
        .sort_values(keys)
        .reset_index(drop=True)
    )
