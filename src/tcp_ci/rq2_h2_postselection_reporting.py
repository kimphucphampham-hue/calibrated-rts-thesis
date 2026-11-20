"""Post-selection H2 comparator reporting for the synthetic Option-A prototype.

Only call with FIXED selection results and separately isolated full-build oracle.
The outcome oracle is never used to generate probabilities, costs or selected IDs.
One Random repeat is not a project-level independent statistical sample.
THIS MODULE IS NOT FINAL/CONFIRMATORY CORE EVIDENCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
import pandas as pd

EVIDENCE_SCOPE = "wp6_2_e3_h2_postselection_prototype"
STATUS_COMPUTED = "computed"
NOT_APPLICABLE_PREFIX = "not_applicable_"
CONDITION_FIELDS = (
    "project", "model", "probability", "rho", "cost_policy", "observation_regime",
)
GROUP_FIELDS = CONDITION_FIELDS + ("baseline", "comparison_role", "repetition")


@dataclass(frozen=True)
class H2PrototypeReports:
    """Build-level rows, all-build status/denominators and project summaries."""

    build_metrics: pd.DataFrame
    build_status: pd.DataFrame
    project_metrics: pd.DataFrame


def _oracle_for_build(oracle: pd.DataFrame, project: str, index: int) -> pd.DataFrame:
    required = {"project_id", "build_id", "build_index", "test_id", "failed", "execution_time_ms"}
    if not isinstance(oracle, pd.DataFrame) or not required.issubset(oracle.columns):
        raise ValueError("H2 oracle requires full canonical identity and post-execution columns.")
    if oracle.empty or oracle[list(required)].isna().any().any():
        raise ValueError("H2 oracle has null/empty required values.")
    idx = pd.to_numeric(oracle["build_index"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(idx).all() or (idx < 1).any()
            or (idx != np.floor(idx)).any()):
        raise ValueError("Invalid H2 oracle build indices.")
    if oracle.duplicated(["project_id", "build_index", "test_id"]).any():
        raise ValueError("Duplicate H2 oracle build/test identity.")
    mask = oracle["project_id"].astype(str).eq(project) & (idx == index)
    frame = oracle.loc[mask].copy()
    if frame.empty or frame["build_id"].astype(str).nunique() != 1:
        raise ValueError("Missing or ambiguous H2 build oracle identity.")
    return frame


def _truth(frame: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    ids = tuple(frame["test_id"].astype(str))
    if len(set(ids)) != len(ids) or any(not i.strip() or "|" in i for i in ids):
        raise ValueError("Invalid H2 oracle test identity.")
    y = pd.to_numeric(frame["failed"], errors="raise").to_numpy(dtype=float)
    times = pd.to_numeric(frame["execution_time_ms"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all()
            or not np.isfinite(times).all() or (times < 0).any()):
        raise ValueError("Invalid H2 oracle outcomes/durations.")
    return ids, y, times


def _require_selection(selected, candidate_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(selected, (tuple, list)):
        raise ValueError("H2 selections must be explicit test-id sequences.")
    ids = tuple(selected)
    if any(not isinstance(s, str) for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("H2 selection contains duplicate/non-string test IDs.")
    if not set(ids).issubset(set(candidate_ids)):
        raise ValueError("H2 selection contains test IDs outside the oracle build.")
    return ids


def _score_one(*, common: dict, baseline: str, role: str, repetition: int,
               selected, estimated_cost_ms: float, cap_ms: float,
               oracle_ids: tuple[str, ...], failures: np.ndarray,
               durations: np.ndarray) -> dict:
    selected_ids = _require_selection(selected, oracle_ids)
    mask = np.array([v in set(selected_ids) for v in oracle_ids], dtype=bool)
    actual_ms = float(np.sum(durations[mask]))
    full_ms = float(np.sum(durations))
    failed_total = int(np.sum(failures))
    captured = int(np.sum(failures[mask]))
    est = float(estimated_cost_ms)
    if not isfinite(est) or est < 0 or not isfinite(cap_ms) or cap_ms <= 0:
        raise ValueError("Invalid H2 estimated cost/cap.")
    if role in ("reference", "budget_comparator") and est > cap_ms + 1e-7:
        raise ValueError("H2 primary comparator exceeded estimated-cost cap.")
    return {
        **common,
        "baseline": baseline,
        "comparison_role": role,
        "repetition": repetition,
        "n_tests": len(oracle_ids),
        "n_selected": len(selected_ids),
        "selected_test_ids": "|".join(sorted(selected_ids)),
        "budget_cap_ms": cap_ms,
        "estimated_selected_ms": est,
        "budget_utilization": est / cap_ms,
        "observed_selected_ms": actual_ms,
        "observed_full_ms": full_ms,
        "tsr": len(selected_ids) / len(oracle_ids),
        "rtr": actual_ms / full_ms if full_ms > 0 else np.nan,
        "rtr_defined": full_ms > 0,
        "total_failures": failed_total,
        "captured_failures": captured,
        "missed_failures": failed_total - captured,
        "failure_recall": captured / failed_total if failed_total else np.nan,
        "missed_failure_rate": (failed_total - captured) / failed_total if failed_total else np.nan,
        "failure_metrics_defined": failed_total > 0,
        "evidence_scope": EVIDENCE_SCOPE,
    }


def report_h2_prototype_steps(steps: Sequence, oracle: pd.DataFrame) -> H2PrototypeReports:
    """Score fixed choices *after* all selections and aggregate by project/repeat.

    Each project/repetition is its own stratum. No p-values are computed;
    users must carry project-level estimates into separately specified inference.
    For non-applicable fallback builds, add status rows, NEVER synthetic budgets.
    """
    if not isinstance(steps, (tuple, list)) or not steps:
        raise ValueError("Nonempty H2 steps required.")
    scored = []
    statuses = []
    seen = set()
    for step in steps:
        if not hasattr(step, "report") or not isinstance(step.report, dict):
            raise ValueError("Malformed H2 step/report.")
        report = step.report
        project = str(step.project_id)
        build_index = int(step.build_index)
        model = str(step.model)
        probability = str(step.probability_variant)
        rho = float(report["rho"])
        if (not isfinite(rho) or rho not in (0.90, 0.95, 0.99)
                or report["project"] != project or report["model"] != model):
            raise ValueError("Invalid H2 step condition/identity.")
        if report["probability"] != {"uncalibrated":"p_uncalibrated", "platt":"p_platt", "isotonic":"p_isotonic"}.get(probability):
            raise ValueError("H2 step probability/report mismatch.")
        condition = (project, model, probability, rho)
        key = condition + (build_index,)
        if key in seen:
            raise ValueError("Duplicate H2 build/condition step.")
        seen.add(key)
        frame = _oracle_for_build(oracle, project, build_index)
        ids, failed, times = _truth(frame)
        if int(report["n_tests"]) != len(ids):
            raise ValueError("H2 scorer/report candidate count mismatch.")
        if (str(report.get("build_id")) != str(frame["build_id"].iloc[0])
                or int(report["build_index"]) != build_index):
            raise ValueError("H2 scoring oracle/report build identity mismatch.")
        if step.h2_status not in ("computed", "not_requested",
                                  "not_applicable_zero_total_predicted_risk",
                                  "not_applicable_no_positive_prior_cost_history"):
            raise ValueError("Unknown H2 selection/reporting status.")
        paired = step.h2_budget_pairing
        if (step.h2_status == STATUS_COMPUTED) != (paired is not None):
            raise ValueError("H2 pairing/status inconsistent.")
        if (step.h2_status.startswith(NOT_APPLICABLE_PREFIX)
                and step.selection_mode != "fallback_retest_all"):
            raise ValueError("H2 not-applicable status requires conservative fallback.")
        status = dict(project=project, model=model, probability=probability,
                      rho=rho, build_index=build_index, build_id=str(frame["build_id"].iloc[0]),
                      h2_status=step.h2_status, n_tests=len(ids),
                      evidence_scope=EVIDENCE_SCOPE)
        statuses.append(status)
        if paired is None:
            continue
        rp = paired.random_pair
        cap = float(rp.reference_estimated_cost_ms)
        expected_condition = (
            "model=" + model, "probability=" + probability,
            "rho=" + format(rho, ".2f"),
            "cost_policy=historical_median", "observation=retrospective_full_information",
        )
        if (rp.project_id != project or rp.build_index != build_index
                or rp.condition_key != expected_condition
                or set(rp.full_retest.selected_test_ids) != set(ids)
                or set(rp.reference_selected_test_ids) != set(step.selected_test_ids)
                or report.get("selection_mode") != "normal_risk_density"
                or step.selection_mode != "normal_risk_density"
                or report.get("cost_policy") != "historical_median"
                or report.get("observation_regime") != "retrospective_full_information"
                or bool(paired.h2_primary_rho) != bool(rho in (0.95, 0.99))):
            raise ValueError("H2 comparator, target and reference selection mismatch.")
        if not np.isclose(float(report["estimated_selected_ms"]), cap, rtol=1e-9, atol=1e-7):
            raise ValueError("H2 reporting reference cost differs from selector.")
        if len(rp.random_replicates) != 30:
            raise ValueError("H2 reporting expects 30 seeded Random replicas.")
        common = dict(project=project, model=model, probability=probability,
                      rho=rho, cost_policy="historical_median",
                      observation_regime="retrospective_full_information",
                      build_index=build_index, build_id=status["build_id"],
                      h2_primary_rho=bool(paired.h2_primary_rho))

        def append(name, role, rep, selection, cost):
            scored.append(_score_one(
                common=common, baseline=name, role=role, repetition=rep,
                selected=selection, estimated_cost_ms=cost, cap_ms=cap,
                oracle_ids=ids, failures=failed, durations=times,
            ))

        append("risk_density_reference", "reference", -1,
               step.selected_test_ids, cap)
        append("failure_history_matched_budget", "budget_comparator", -1,
               paired.failure_history_budget.selected_test_ids,
               paired.failure_history_budget.estimated_cost_ms)
        append("history_duration_matched_budget", "budget_comparator", -1,
               paired.history_duration_budget.selected_test_ids,
               paired.history_duration_budget.estimated_cost_ms)
        append("retest_all", "supplementary_full_retest", -1,
               rp.full_retest.selected_test_ids, rp.full_retest.estimated_cost_ms)
        seen_reps = set()
        for item in rp.random_replicates:
            if item.repetition in seen_reps or not 0 <= item.repetition < 30:
                raise ValueError("Invalid/duplicate H2 Random repetition.")
            seen_reps.add(item.repetition)
            if item.budget_capped is None:
                raise ValueError("Missing H2 Random budget comparator.")
            append("random_matched_budget", "budget_comparator", item.repetition,
                   item.budget_capped.selected_test_ids, item.budget_capped.estimated_cost_ms)
            append("random_matched_count", "supplementary_matched_count", item.repetition,
                   item.matched_count.selected_test_ids, item.matched_count.estimated_cost_ms)
        if seen_reps != set(range(30)):
            raise ValueError("Incomplete H2 Random repetitions.")

    status_frame = pd.DataFrame(statuses)
    build_frame = pd.DataFrame(scored)
    if build_frame.empty:
        return H2PrototypeReports(build_frame, status_frame, pd.DataFrame())
    if build_frame.duplicated(list(GROUP_FIELDS) + ["build_index"]).any():
        raise ValueError("Duplicate H2 build/baseline/repetition rows.")
    project_rows = []
    for key, frame in build_frame.groupby(list(GROUP_FIELDS), sort=True):
        full_ms = float(frame["observed_full_ms"].sum())
        selected_ms = float(frame["observed_selected_ms"].sum())
        failed_total = int(frame["total_failures"].sum())
        missed = int(frame["missed_failures"].sum())
        matching_status = status_frame[
            status_frame["project"].eq(key[0])
            & status_frame["model"].eq(key[1])
            & status_frame["probability"].eq(key[2])
            & status_frame["rho"].eq(key[3])
        ]
        project_rows.append({
            **dict(zip(GROUP_FIELDS, key)),
            "n_input_builds": len(matching_status),
            "n_h2_computed_builds": len(frame),
            "n_not_applicable_builds": int(matching_status["h2_status"].str.startswith(NOT_APPLICABLE_PREFIX).sum()),
            "n_not_requested_builds": int(matching_status["h2_status"].eq("not_requested").sum()),
            "n_undefined_rtr_builds": int((~frame["rtr_defined"]).sum()),
            "n_zero_failure_builds": int((~frame["failure_metrics_defined"]).sum()),
            "n_tests": int(frame["n_tests"].sum()),
            "n_selected": int(frame["n_selected"].sum()),
            "tsr_micro": float(frame["n_selected"].sum() / frame["n_tests"].sum()),
            "observed_selected_ms": selected_ms,
            "observed_full_ms": full_ms,
            "rtr_micro": selected_ms / full_ms if full_ms > 0 else np.nan,
            "total_failures": failed_total,
            "captured_failures": int(frame["captured_failures"].sum()),
            "missed_failures": missed,
            "failure_recall_micro": (failed_total - missed) / failed_total if failed_total else np.nan,
            "missed_failure_rate_micro": missed / failed_total if failed_total else np.nan,
            "estimated_selected_ms": float(frame["estimated_selected_ms"].sum()),
            "budget_cap_ms": float(frame["budget_cap_ms"].sum()),
            "budget_utilization": float(frame["estimated_selected_ms"].sum() / frame["budget_cap_ms"].sum()),
            "evidence_scope": EVIDENCE_SCOPE,
        })
    project_frame = pd.DataFrame(project_rows).sort_values(list(GROUP_FIELDS)).reset_index(drop=True)
    return H2PrototypeReports(
        build_metrics=build_frame.sort_values(list(GROUP_FIELDS) + ["build_index"]).reset_index(drop=True),
        build_status=status_frame.sort_values(["project", "model", "probability", "rho", "build_index"]).reset_index(drop=True),
        project_metrics=project_frame,
    )
