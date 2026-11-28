"""Experimental fitted RQ1 snapshot: fit once, predict only from later-build context.

Pre-freeze engineering artifact. Not an approved static/periodic RQ3 schedule.
The code deliberately reuses frozen RQ1 model/calibration helpers without editing
existing pipeline/Protocol v1. An object is held in memory, not serialized.
"""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.tcp_ci.rq1_pipeline import (
    PROBABILITY_COLUMNS,
    build_isotonic_calibrator,
    build_platt_calibrator,
    build_rq1_model,
    fit_probability_model,
    materialize_calibration_partitions,
    positive_probability,
    split_development_for_calibration,
    validate_rq1_integration_contract,
)


TARGET_ID_COLUMNS = ("project_id", "build_id", "build_index", "test_id")
TARGET_PROBABILITY_COLUMNS = tuple(PROBABILITY_COLUMNS[name] for name in ("uncalibrated", "platt", "isotonic"))


@dataclass(frozen=True)
class RQ1FittedSnapshot:
    """Model and calibrators fitted using a fixed chronological development cohort.

    A frozen dataclass does not render sklearn estimator objects immutable.
    Callers must not mutate fitted_model/platt/isotonic after construction.
    """

    model_name: str
    project_id: str
    feature_columns: Tuple[str, ...]
    fitted_model: Any
    platt: Optional[Any]
    isotonic: Optional[Any]
    calibration_support_sufficient: bool
    last_development_build_index: int
    model_fit_last_build_index: int
    calibration_first_build_index: Optional[int]
    calibration_last_build_index: Optional[int]


def _canonical_build_indices(frame: pd.DataFrame, label: str) -> np.ndarray:
    if "build_index" not in frame.columns:
        raise ValueError(f"{label}: missing build_index")
    values = pd.to_numeric(frame["build_index"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values == np.floor(values)).all() or (values < 1).any():
        raise ValueError(f"{label}: build_index must be positive finite integers")
    return values.astype(np.int64)


def _validate_rows(frame: pd.DataFrame, label: str, *, labels_expected: bool) -> tuple:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label}: expected non-empty DataFrame")
    required = set(TARGET_ID_COLUMNS)
    if labels_expected:
        required.add("failed")
    if required - set(frame.columns):
        raise ValueError(f"{label}: missing columns {sorted(required - set(frame.columns))}")
    if frame[list(TARGET_ID_COLUMNS)].isna().any().any():
        raise ValueError(f"{label}: missing row identity")
    if frame.duplicated(list(TARGET_ID_COLUMNS)).any():
        raise ValueError(f"{label}: duplicate row identity")
    projects = frame["project_id"].astype(str).unique()
    if len(projects) != 1:
        raise ValueError(f"{label}: exactly one project is required")
    indices = _canonical_build_indices(frame, label)
    if labels_expected:
        labels = pd.to_numeric(frame["failed"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(f"{label}: failed must be binary 0/1")
    return str(projects[0]), indices


def fit_rq1_snapshot(
    development: pd.DataFrame,
    *,
    model_name: str,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
    timing_sink=None,
) -> RQ1FittedSnapshot:
    """Fit model/optional calibrators once; do not consume any evaluation row.

    Split/cohort policies are delegated to frozen RQ1 implementation.
    Insufficient calibration support retains only uncalibrated predictions.
    """
    clock = perf_counter_ns if timing_sink is not None else lambda: 0
    start = clock()
    project, indices = _validate_rows(development, "development", labels_expected=True)
    features = tuple(validate_rq1_integration_contract(
        rq1_config=rq1_config,
        model_config=model_config,
        calibration_config=calibration_config,
        core_features_config=core_features_config,
    ))
    if not features or any(name not in development.columns for name in features):
        raise ValueError("development: missing required RQ1 features")
    split = split_development_for_calibration(
        development,
        rq1_config=rq1_config,
        calibration_config=calibration_config,
        build_column="build_index",
        target_column="failed",
    )
    model_fit, calibration = materialize_calibration_partitions(
        development, split, build_column="build_index"
    )
    if model_fit.empty:
        raise ValueError("model_fit partition is empty")
    fit_indices = _canonical_build_indices(model_fit, "model_fit")
    calibration_min = calibration_max = None
    if not calibration.empty:
        cal_indices = _canonical_build_indices(calibration, "calibration")
        calibration_min, calibration_max = int(cal_indices.min()), int(cal_indices.max())
        if int(fit_indices.max()) >= calibration_min:
            raise ValueError("model_fit/calibration chronology violation")
    if int(fit_indices.max()) > int(indices.max()):
        raise ValueError("model_fit exceeds development boundary")
    sufficient = bool(split.sufficient)
    if sufficient and calibration.empty:
        raise ValueError("sufficient calibration split unexpectedly empty")

    model = build_rq1_model(
        model_name=model_name, feature_columns=list(features), model_config=model_config
    )
    t = clock()
    fitted_model = fit_probability_model(
        model,
        model_fit[list(features)],
        model_fit["failed"],
    )
    model_fit_ns = clock() - t
    calibration_prediction_ns = platt_fit_ns = isotonic_fit_ns = 0
    platt = isotonic = None
    if sufficient:
        t = clock()
        calibration_probability = positive_probability(fitted_model, calibration[list(features)])
        calibration_prediction_ns = clock() - t
        calibration_target = calibration["failed"]
        platt = build_platt_calibrator(calibration_config)
        t = clock()
        platt.fit(calibration_probability, calibration_target)
        platt_fit_ns = clock() - t
        isotonic = build_isotonic_calibrator(calibration_config)
        t = clock()
        isotonic.fit(calibration_probability, calibration_target)
        isotonic_fit_ns = clock() - t

    result = RQ1FittedSnapshot(
        model_name=model_name,
        project_id=project,
        feature_columns=features,
        fitted_model=fitted_model,
        platt=platt,
        isotonic=isotonic,
        calibration_support_sufficient=sufficient,
        last_development_build_index=int(indices.max()),
        model_fit_last_build_index=int(fit_indices.max()),
        calibration_first_build_index=calibration_min,
        calibration_last_build_index=calibration_max,
    )

    total = clock() - start
    if timing_sink is not None:
        timing_sink(dict(model=model_name,cutoff=result.last_development_build_index,
            snapshot_total_ns=total,model_fit_ns=model_fit_ns,
            calibration_prediction_ns=calibration_prediction_ns,platt_fit_ns=platt_fit_ns,
            isotonic_fit_ns=isotonic_fit_ns,
            other_snapshot_ns=total-model_fit_ns-calibration_prediction_ns-platt_fit_ns-isotonic_fit_ns,
            model_fit_builds=len(split.model_fit_builds),calibration_builds=len(split.calibration_builds),
            expanded_by_builds=split.expanded_by_builds,calibration_status=split.status,
            positive_rows=split.support.positive_rows,negative_rows=split.support.negative_rows,
            failing_builds=split.support.failing_builds,passing_builds=split.support.passing_builds))
    return result


def predict_from_rq1_snapshot(
    snapshot: RQ1FittedSnapshot,
    target_preexecution: pd.DataFrame,
) -> pd.DataFrame:
    """Score later target without fitting and without accessing target oracle.

    Reject all post-execution columns and all unrecognized input fields rather
    than dropping potentially leaked target information silently.
    """
    if not isinstance(snapshot, RQ1FittedSnapshot):
        raise TypeError("snapshot must be RQ1FittedSnapshot")
    project, indices = _validate_rows(target_preexecution, "target", labels_expected=False)
    permitted = set(TARGET_ID_COLUMNS) | set(snapshot.feature_columns)
    unknown = set(target_preexecution.columns) - permitted
    if unknown:
        raise ValueError(f"target: unexpected/unsafe columns: {sorted(unknown)}")
    if project != snapshot.project_id:
        raise ValueError("target belongs to another project")
    if (indices <= snapshot.last_development_build_index).any():
        raise ValueError("target is not strictly after the snapshot's development")
    if len(set(indices.tolist())) != 1:
        raise ValueError("target must contain exactly one build")
    missing_features = set(snapshot.feature_columns) - set(target_preexecution.columns)
    if missing_features:
        raise ValueError(f"target: missing features: {sorted(missing_features)}")
    base = np.asarray(
        positive_probability(snapshot.fitted_model, target_preexecution[list(snapshot.feature_columns)]),
        dtype=float,
    )
    if base.shape != (len(target_preexecution),) or not np.isfinite(base).all():
        raise ValueError("invalid RQ1 base probability array")
    if ((base < 0) | (base > 1)).any():
        raise ValueError("base probabilities outside [0,1]")
    predictions = target_preexecution[list(TARGET_ID_COLUMNS)].copy()
    predictions[PROBABILITY_COLUMNS["uncalibrated"]] = base
    if snapshot.calibration_support_sufficient:
        if snapshot.platt is None or snapshot.isotonic is None:
            raise ValueError("calibrators missing from sufficient snapshot")
        predictions[PROBABILITY_COLUMNS["platt"]] = snapshot.platt.transform(base)
        predictions[PROBABILITY_COLUMNS["isotonic"]] = snapshot.isotonic.transform(base)
    else:
        predictions[PROBABILITY_COLUMNS["platt"]] = float("nan")
        predictions[PROBABILITY_COLUMNS["isotonic"]] = float("nan")
    return predictions[list(TARGET_ID_COLUMNS) + list(TARGET_PROBABILITY_COLUMNS)]
