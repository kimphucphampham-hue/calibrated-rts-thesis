"""Synthetic-only single-build RQ1 -> RQ2 integration (Option A candidate).

The target's true outcomes are quarantined in an offline scoring oracle.
RQ1 sees only a pre-execution feature view with a structural dummy label.
Historical observations are earlier by build index under the *assumed*
retrospective full-information convention, NOT verified CI finish timestamps.

NOT a final longitudinal runner, Method Freeze, or deployment simulator.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq1_pipeline import PROBABILITY_COLUMNS
from src.tcp_ci.rq1_preselection_adapter import predict_preselection_rq1
from src.tcp_ci.rq1_fitted_snapshot_prototype import predict_from_rq1_snapshot
from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback
from src.tcp_ci.rq2_cold_start_reporting import evaluate_post_selection
from src.tcp_ci.rq2_cost_adapter import estimate_build_costs
from src.tcp_ci.rq2_h2_budget_pairing import (
    H2BudgetPairedBuild, pair_h2_budget_comparators_for_build,
)

IDENTITY = ("project_id", "build_id", "build_index", "test_id")
POST_EXECUTION = frozenset({"failed", "execution_time_ms", "verdict_raw", "duration"})
REGIME = "retrospective_full_information"


@dataclass(frozen=True)
class SingleBuildResult:
    project_id: str
    build_index: int
    model: str
    probability_variant: str
    selected_test_ids: Tuple[str, ...]
    selection_mode: str
    prediction_view: pd.DataFrame
    rq1_diagnostics: Dict
    report: Dict
    h2_budget_pairing: Optional[H2BudgetPairedBuild] = None
    h2_status: str = "not_requested"


def _validate_build(frame: pd.DataFrame, name: str) -> Tuple[str, str, int, set]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    missing = set(IDENTITY) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing identity columns: {sorted(missing)}")
    for column in ("project_id", "build_id", "test_id"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has invalid {column}.")
    project = frame["project_id"].astype(str)
    build_id = frame["build_id"].astype(str)
    indices = pd.to_numeric(frame["build_index"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or (indices < 1).any() or (indices != np.floor(indices)).any():
        raise ValueError(f"{name} has invalid build_index.")
    if project.nunique() != 1 or build_id.nunique() != 1 or len(set(indices)) != 1:
        raise ValueError(f"{name} must contain exactly one project/build.")
    test_ids = frame["test_id"].astype(str)
    if test_ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate candidate test IDs.")
    return project.iloc[0], build_id.iloc[0], int(indices[0]), set(test_ids)


def _validate_history(history: pd.DataFrame, project: str, target_index: int) -> None:
    required = {"project_id", "build_index", "test_id", "execution_time_ms"}
    if not isinstance(history, pd.DataFrame) or not required.issubset(history.columns):
        raise ValueError("History missing required cost provenance columns.")
    if history.empty:
        return
    if history["project_id"].isna().any() or (history["project_id"].astype(str) != project).any():
        raise ValueError("History must contain only the target project.")
    indices = pd.to_numeric(history["build_index"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or (indices < 1).any() or (indices != np.floor(indices)).any():
        raise ValueError("History has invalid build_index.")
    if (indices >= target_index).any():
        raise ValueError("History contains target/future build results.")
    if history["test_id"].isna().any() or history["test_id"].astype(str).str.strip().eq("").any():
        raise ValueError("History has invalid test IDs.")
    if history.duplicated(["project_id", "build_index", "test_id"]).any():
        raise ValueError("History has duplicate build/test results.")
    costs = pd.to_numeric(history["execution_time_ms"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("History contains invalid realized durations.")


def evaluate_one_build_with_rq1(
    development: pd.DataFrame,
    target_preexecution: pd.DataFrame,
    history: pd.DataFrame,
    target_oracle: pd.DataFrame,
    model_name: str,
    probability_variant: str,
    rho: float,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
    *,
    rq1_snapshot=None,  # opt-in synthetic fitted object; legacy default unchanged
    include_h2_budget: bool = False,  # opt-in causal H2 selection comparison
) -> SingleBuildResult:
    """Bridge existing RQ1 adapter to real RQ2 selector/reporting for ONE build.

    `target_preexecution` MUST NOT contain realized target outcomes/costs.
    `target_oracle` contains full recorded truth and is not touched until
    a selection is fixed. Callers must ensure upstream F1/F2/F3 features
    and prior historical state are causal; this function cannot certify them.
    """
    if not isinstance(include_h2_budget, bool):
        raise ValueError("include_h2_budget must be boolean.")
    project, build_id, target_index, expected_ids = _validate_build(
        target_preexecution, "target_preexecution"
    )
    forbidden = POST_EXECUTION.intersection(target_preexecution.columns)
    if forbidden:
        raise ValueError(f"Preexecution input contains post-execution columns: {sorted(forbidden)}")
    if "execution_time_mean_history_ms" not in target_preexecution.columns:
        raise ValueError("Preexecution input lacks execution_time_mean_history_ms.")
    oracle_project, oracle_build_id, oracle_index, oracle_ids = _validate_build(
        target_oracle, "target_oracle"
    )
    if (project, build_id, target_index, expected_ids) != (
        oracle_project, oracle_build_id, oracle_index, oracle_ids
    ):
        raise ValueError("Target and scoring oracle build/test identities do not match.")
    if not {"failed", "execution_time_ms"}.issubset(target_oracle.columns):
        raise ValueError("Scoring oracle needs full failed and execution_time_ms columns.")
    # Inspect truth only to validate its post-selection scoring schema; never
    # let its values participate in prediction, cost estimation, or selection.
    _validate_history(history, project, target_index)
    if probability_variant not in PROBABILITY_COLUMNS:
        raise ValueError("Unsupported RQ1 probability variant.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Invalid model name.")

    # WP5.17.16: keep legacy refit path as default; snapshot path is opt-in.
    if rq1_snapshot is None:
        # This adapter validates strictly earlier development and masks target labels.
        predictions, diagnostics = predict_preselection_rq1(
            development=development,
            target_preexecution=target_preexecution,
            model_name=model_name,
            rq1_config=rq1_config,
            model_config=model_config,
            calibration_config=calibration_config,
            core_features_config=core_features_config,
        )
    else:
        if (rq1_snapshot.model_name != model_name
                or str(rq1_snapshot.project_id) != str(project)):
            raise ValueError('Snapshot model/project identity mismatch.')
        safe_columns = list(dict.fromkeys(list(IDENTITY) + list(rq1_snapshot.feature_columns)))
        if POST_EXECUTION.intersection(safe_columns):
            raise ValueError('Snapshot features include post-execution fields.')
        missing = set(safe_columns).difference(target_preexecution.columns)
        if missing:
            raise ValueError(f'Snapshot target features missing: {sorted(missing)}')
        # This whitelist excludes oracle fields AND unrelated columns.
        predictions = predict_from_rq1_snapshot(
            rq1_snapshot, target_preexecution.loc[:, safe_columns].copy(deep=True)
        )
        diagnostics = {
            'prediction_mode': 'fixed_initial_snapshot_synthetic',
            'snapshot_last_development_build_index': rq1_snapshot.last_development_build_index,
            'model_fit_last_build_index': rq1_snapshot.model_fit_last_build_index,
            'calibration_support_sufficient': rq1_snapshot.calibration_support_sufficient,
        }

    col = PROBABILITY_COLUMNS[probability_variant]
    probability = pd.to_numeric(predictions[col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        # Calibrator skip (entirely NaN) needs explicit status in final runner.
        raise ValueError("Requested probability variant unavailable or invalid; do not silently skip.")

    aligned = target_preexecution.loc[:, list(IDENTITY)].merge(
        predictions.loc[:, list(IDENTITY) + [col]],
        on=list(IDENTITY), how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if len(aligned) != len(target_preexecution) or not aligned["_merge"].eq("both").all():
        raise ValueError("Missing or extra RQ1 predictions for the target build.")
    if aligned[col].isna().any():
        raise ValueError("Missing RQ1 probabilities for selected variant.")
    # Construct cost inputs solely from decision-time feature values.
    decision = target_preexecution.loc[:, [
        "project_id", "build_index", "test_id", "execution_time_mean_history_ms"
    ]].copy(deep=True)
    aligned_p = pd.to_numeric(aligned[col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(aligned_p).all() or ((aligned_p < 0) | (aligned_p > 1)).any():
        raise ValueError("Invalid aligned probabilities.")

    selection = select_with_cold_start_fallback(
        target=decision, history=history.copy(deep=True), probabilities=aligned_p, rho=rho
    )
    # WP6.2-E2: optional PRE-SELECTION comparison using causally materialized
    # F1 history rates plus the SAME historical cost adapter as the selector.
    # No target oracle value is consulted before the compared sets are fixed.
    h2 = None
    h2_status = "not_requested"
    if include_h2_budget:
        if selection.selection_mode == "fallback_retest_all":
            if selection.fallback_reason not in {
                "zero_total_predicted_risk", "no_positive_prior_cost_history",
            }:
                raise ValueError("Unsupported H2 fallback reason.")
            # No defensible positive reference estimated-cost cap exists here.
            h2_status = "not_applicable_" + selection.fallback_reason
        elif selection.selection_mode == "normal_risk_density":
            required = {"failure_rate_history", "prior_execution_count"}
            if not required.issubset(target_preexecution.columns):
                raise ValueError("H2 requires causally materialized F1 history features.")
            counts = pd.to_numeric(
                target_preexecution["prior_execution_count"], errors="raise"
            ).to_numpy(dtype=float)
            rates = pd.to_numeric(
                target_preexecution["failure_rate_history"], errors="raise"
            ).to_numpy(dtype=float)
            if (not np.isfinite(counts).all() or (counts < 0).any()
                    or (counts != np.floor(counts)).any()):
                raise ValueError("Invalid H2 historical execution counts.")
            unseen = counts == 0
            # Explicit new-test convention: absent prior failures => rate 0.
            # Only an unseen test may have a missing historical failure rate.
            if (np.isinf(rates).any() or np.isnan(rates[~unseen]).any()
                    or ((rates < 0) | (rates > 1)).any()
                    or (np.isfinite(rates[unseen]) & (rates[unseen] != 0)).any()):
                raise ValueError("Invalid or inconsistent H2 historical failure rates.")
            rates[unseen] = 0.0

            estimated = estimate_build_costs(
                target=decision, history=history.copy(deep=True),
                zero_cost_floor_ms=None,
            )
            ids = tuple(str(value) for value in decision["test_id"].tolist())
            if (not estimated.index.equals(decision.index)
                    or tuple(str(x) for x in estimated["test_id"]) != ids):
                raise RuntimeError("H2 estimated-cost alignment changed.")
            costs_ms = estimated["estimated_cost_ms"].to_numpy(dtype=float)
            chosen_set = set(selection.selected_test_ids)
            reference_cost = float(np.sum([
                cost for test_id, cost in zip(ids, costs_ms)
                if test_id in chosen_set
            ]))
            if (selection.estimated_selected_cost_ms is None
                    or not np.isclose(
                        reference_cost, selection.estimated_selected_cost_ms,
                        rtol=1e-9, atol=1e-7,
                    )):
                raise RuntimeError("H2 reference/selector cost estimates differ.")
            condition = (
                "model=" + model_name,
                "probability=" + probability_variant,
                "rho=" + format(float(rho), ".2f"),
                "cost_policy=historical_median",
                "observation=" + REGIME,
            )
            h2 = pair_h2_budget_comparators_for_build(
                project_id=project, build_index=target_index,
                condition_key=condition, test_ids=ids,
                estimated_costs_ms=costs_ms,
                historical_failure_rates=rates,
                reference_selected_test_ids=selection.selected_test_ids,
            )
            h2_status = "computed"
        else:
            raise ValueError("Unsupported H2 selection mode.")

    # Only after selection AND all H2 compared sets are fixed does target truth
    # reach the retrospective scoring function.
    report = evaluate_post_selection(
        target=target_oracle.copy(deep=True), selection=selection, rho=rho,
        model=model_name, probability=col, cost_policy="historical_median",
        baseline="risk_density", observation_regime=REGIME,
    )
    return SingleBuildResult(
        project_id=project, build_index=target_index, model=model_name,
        probability_variant=probability_variant,
        selected_test_ids=tuple(selection.selected_test_ids),
        selection_mode=str(selection.selection_mode),
        prediction_view=predictions.copy(deep=True),
        rq1_diagnostics=dict(diagnostics), report=dict(report),
        h2_budget_pairing=h2, h2_status=h2_status,
    )
