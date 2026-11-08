"""Synthetic-only single-build RQ1 -> RQ2 integration (Option A candidate).

The target's true outcomes are quarantined in an offline scoring oracle.
RQ1 sees only a pre-execution feature view with a structural dummy label.
Historical observations are earlier by build index under the *assumed*
retrospective full-information convention, NOT verified CI finish timestamps.

NOT a final longitudinal runner, Method Freeze, or deployment simulator.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq1_pipeline import PROBABILITY_COLUMNS
from src.tcp_ci.rq1_preselection_adapter import predict_preselection_rq1
from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback
from src.tcp_ci.rq2_cold_start_reporting import evaluate_post_selection

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
) -> SingleBuildResult:
    """Bridge existing RQ1 adapter to real RQ2 selector/reporting for ONE build.

    `target_preexecution` MUST NOT contain realized target outcomes/costs.
    `target_oracle` contains full recorded truth and is not touched until
    a selection is fixed. Callers must ensure upstream F1/F2/F3 features
    and prior historical state are causal; this function cannot certify them.
    """
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
    # Only after selection is computed does target truth reach the offline scorer.
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
    )
