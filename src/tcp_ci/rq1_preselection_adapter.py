"""Preselection-only RQ1 adapter for an UNAPPROVED Option A prototype.

Inputs must contain causally precomputed RQ1 features. The actual target
failure labels and realized costs are never passed to run_rq1_model.
The RQ1 helper still receives a structurally required *dummy* failed=0
field in evaluation; this is NOT the recorded target label.

This adapter neither builds historical features nor certifies wall-clock
completion/observability of earlier builds. It is not a final runner.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq1_pipeline import (
    PROBABILITY_COLUMNS,
    run_rq1_model,
    validate_rq1_integration_contract,
)

IDENTITY_COLUMNS = ("project_id", "build_id", "build_index", "test_id")
FORBIDDEN_FEATURES = frozenset(
    {"failed", "execution_time_ms", "verdict_raw", "duration"}
)


def _validate_identity(frame: pd.DataFrame, name: str) -> Tuple[str, int]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    missing = set(IDENTITY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing identity columns: {sorted(missing)}")
    for column in ("project_id", "build_id", "test_id"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has invalid {column}.")
    indices = pd.to_numeric(frame["build_index"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or (indices < 1).any() or (indices != np.floor(indices)).any():
        raise ValueError(f"{name} has invalid build_index.")
    projects = frame["project_id"].astype(str).unique()
    if len(projects) != 1:
        raise ValueError(f"{name} must contain exactly one project.")
    keys = list(IDENTITY_COLUMNS)
    if frame.duplicated(keys).any() or frame.duplicated(["project_id", "build_index", "test_id"]).any():
        raise ValueError(f"{name} contains duplicate build-test identities.")
    if frame.groupby("build_index")["build_id"].nunique().gt(1).any():
        raise ValueError(f"{name} maps one index to multiple build IDs.")
    if frame.groupby("build_id")["build_index"].nunique().gt(1).any():
        raise ValueError(f"{name} maps one build ID to multiple indices.")
    return str(projects[0]), int(indices.max())


def predict_preselection_rq1(
    development: pd.DataFrame,
    target_preexecution: pd.DataFrame,
    model_name: str,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> Tuple[pd.DataFrame, Dict]:
    """Predict one target build without exposing its labels/costs to RQ1.

    Returns ID-aligned ``p_uncalibrated``/``p_platt``/``p_isotonic`` only,
    plus RQ1 diagnostics. Unavailable calibration variants remain all-NaN.
    Callers MUST NOT pass RQ1 output diagnostics into RQ2 selection.
    """
    project, target_index = _validate_identity(target_preexecution, "target")
    other_project, max_development_index = _validate_identity(development, "development")
    if project != other_project:
        raise ValueError("Development and target must have the same project.")
    indices = pd.to_numeric(target_preexecution["build_index"], errors="raise")
    if indices.nunique() != 1:
        raise ValueError("Target must contain exactly one build_index.")
    if target_preexecution["build_id"].nunique() != 1:
        raise ValueError("Target must contain exactly one build_id.")
    if max_development_index >= target_index:
        raise ValueError("Development includes current/future target build.")
    if "failed" not in development.columns:
        raise ValueError("Development requires historical failed labels.")
    y = pd.to_numeric(development["failed"], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Historical labels must be binary and nonmissing.")

    configured_features = validate_rq1_integration_contract(
        rq1_config=rq1_config,
        model_config=model_config,
        calibration_config=calibration_config,
        core_features_config=core_features_config,
    )
    if isinstance(configured_features, str):
        raise ValueError("RQ1 feature contract returned an invalid feature list.")
    features = list(configured_features)
    if not features or len(features) != len(set(features)) or not all(
        isinstance(c, str) and c.strip() for c in features
    ):
        raise ValueError("Invalid RQ1 feature contract.")
    if any(feature.lower() in FORBIDDEN_FEATURES for feature in features):
        raise ValueError("RQ1 feature contract contains post-execution fields.")
    for name, frame in (("development", development), ("target", target_preexecution)):
        missing = set(features).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing RQ1 pre-execution features: {sorted(missing)}")

    # Whitelists: NEVER read target failed/execution_time_ms, even if present.
    development_view = development.loc[:, list(dict.fromkeys(
        [*IDENTITY_COLUMNS, *features, "failed"]
    ))].copy(deep=True)
    evaluation_view = target_preexecution.loc[:, list(dict.fromkeys(
        [*IDENTITY_COLUMNS, *features]
    ))].copy(deep=True)
    evaluation_view["failed"] = 0  # a structural dummy, NOT target ground truth

    prediction_frame, diagnostics = run_rq1_model(
        development=development_view,
        evaluation=evaluation_view,
        model_name=model_name,
        rq1_config=rq1_config,
        model_config=model_config,
        calibration_config=calibration_config,
        core_features_config=core_features_config,
    )
    if not isinstance(prediction_frame, pd.DataFrame):
        raise ValueError("RQ1 pipeline returned invalid predictions.")
    probability_columns = list(PROBABILITY_COLUMNS.values())
    required = list(IDENTITY_COLUMNS) + probability_columns
    missing_predictions = set(required).difference(prediction_frame.columns)
    if missing_predictions:
        raise ValueError(f"RQ1 predictions missing columns: {sorted(missing_predictions)}")

    # Match by canonical ID; never assume the model preserves output row order.
    target_keys = target_preexecution.loc[:, list(IDENTITY_COLUMNS)].copy(deep=True)
    result = target_keys.merge(
        prediction_frame.loc[:, required],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if len(result) != len(target_keys) or not result["_merge"].eq("both").all():
        raise ValueError("RQ1 predictions are missing target candidates.")
    if len(prediction_frame) != len(target_keys):
        raise ValueError("RQ1 produced unexpected extra prediction rows.")
    result = result.drop(columns="_merge")
    if "uncalibrated" not in PROBABILITY_COLUMNS:
        raise ValueError("RQ1 uncalibrated probability column is unavailable.")

    for variant, column in PROBABILITY_COLUMNS.items():
        try:
            numeric = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RQ1 {variant} returned nonnumeric probabilities.") from exc
        missing_mask = np.isnan(numeric)
        if variant == "uncalibrated" and missing_mask.any():
            raise ValueError("Uncalibrated probabilities cannot be missing.")
        if missing_mask.any() and not missing_mask.all():
            raise ValueError(f"RQ1 {variant} has partially missing probabilities.")
        present = numeric[~missing_mask]
        if not np.isfinite(present).all() or ((present < 0) | (present > 1)).any():
            raise ValueError(f"RQ1 {variant} has invalid probability values.")
        result[column] = numeric
    if not isinstance(diagnostics, dict):
        raise ValueError("RQ1 diagnostics must be a dictionary.")
    return result, dict(diagnostics)
