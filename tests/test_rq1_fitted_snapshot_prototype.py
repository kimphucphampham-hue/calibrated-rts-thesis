"""Synthetic contract tests for pre-freeze RQ1 fitted snapshot (not CORE results)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci.rq1_pipeline import PROBABILITY_COLUMNS, load_yaml, run_rq1_model, validate_rq1_integration_contract
from src.tcp_ci.rq1_fitted_snapshot_prototype import (
    RQ1FittedSnapshot,
    fit_rq1_snapshot,
    predict_from_rq1_snapshot,
)


@pytest.fixture(scope="module")
def cfg():
    return {
        "rq1_config": load_yaml(Path("configs/rq1_v1.yaml")),
        "model_config": load_yaml(Path("configs/rq1_models_v1.yaml")),
        "calibration_config": load_yaml(Path("configs/calibration_v1.yaml")),
        "core_features_config": load_yaml(Path("configs/core_features_v1.yaml")),
    }


@pytest.fixture(scope="module")
def synthetic_data(cfg):
    features = tuple(validate_rq1_integration_contract(**cfg))
    test_ids = ("alpha", "beta", "gamma", "delta")
    project = "snapshot-synthetic"

    def make_features(build_index, test_pos):
        values = {}
        for j, field in enumerate(features):
            key = field.lower()
            if key == "build_index":
                value = int(build_index)
            elif key == "failure_rate_history":
                value = float(((build_index - 1) + test_pos) % 10) / 10.0
            elif key == "recent_failure":
                value = int((build_index + test_pos) % 5 == 0)
            elif key == "execution_time_mean_history_ms":
                value = float(10 + test_pos * 8)
            elif key == "test_age_builds":
                value = int(build_index - 1)
            elif "since_last_failure" in key:
                value = int((build_index + test_pos) % 12)
            elif "consecutive_failures" in key:
                value = int((build_index + test_pos) % 3)
            elif "consecutive_passes" in key:
                value = int((build_index + 2 * test_pos) % 4)
            elif "changed_entity" in key:
                value = int((build_index + test_pos) % 5)
            else:
                value = float(((build_index * (j + 3) + test_pos * 7) % 23) / 11.0)
            values[field] = value
        return values

    development_rows = []
    for build_index in range(1, 201):
        for j, test_id in enumerate(test_ids):
            failed = int(
                (j == 0 and build_index % 3 == 0)
                or (j == 1 and build_index % 5 == 0)
                or (j == 2 and build_index % 11 == 0)
            )
            development_rows.append({
                "project_id": project,
                "build_id": f"b{build_index:04d}",
                "build_index": build_index,
                "test_id": test_id,
                "failed": failed,
                **make_features(build_index, j),
            })
    target_rows = [{
        "project_id": project,
        "build_id": "b0201",
        "build_index": 201,
        "test_id": test_id,
        **make_features(201, j),
    } for j, test_id in enumerate(test_ids)]
    return pd.DataFrame(development_rows), pd.DataFrame(target_rows)


def _fit(cfg, dev):
    return fit_rq1_snapshot(dev, model_name="logistic_regression", **cfg)


def _legacy(cfg, dev, target):
    evaluation = target.copy(deep=True)
    evaluation["failed"] = 0  # Legacy RQ1 validation only; no target oracle data.
    predictions, diagnostics = run_rq1_model(
        development=dev,
        evaluation=evaluation,
        model_name="logistic_regression",
        **cfg,
    )
    return predictions, diagnostics


def _assert_prediction_parity(legacy, observed):
    assert set(observed.columns) == set(["project_id", "build_id", "build_index", "test_id"]) | set(PROBABILITY_COLUMNS.values())
    assert "failed" not in observed.columns
    id_fields = ["project_id", "build_id", "build_index", "test_id"]
    pd.testing.assert_frame_equal(
        legacy[id_fields].reset_index(drop=True),
        observed[id_fields].reset_index(drop=True),
    )
    for column in PROBABILITY_COLUMNS.values():
        np.testing.assert_allclose(
            legacy[column].to_numpy(dtype=float),
            observed[column].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-10, equal_nan=True,
        )


def test_real_logistic_sufficient_calibration_parity(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    assert isinstance(snapshot, RQ1FittedSnapshot)
    assert snapshot.calibration_support_sufficient
    assert snapshot.platt is not None and snapshot.isotonic is not None
    assert snapshot.model_fit_last_build_index < snapshot.calibration_first_build_index
    assert snapshot.calibration_last_build_index <= snapshot.last_development_build_index
    legacy, diagnostics = _legacy(cfg, dev, target)
    assert diagnostics.get("calibration_support_sufficient") is True
    observed = predict_from_rq1_snapshot(snapshot, target)
    _assert_prediction_parity(legacy, observed)


def test_repeated_scoring_does_not_refit(cfg, synthetic_data, monkeypatch):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    import src.tcp_ci.rq1_fitted_snapshot_prototype as module

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("snapshot prediction attempted to fit")
    monkeypatch.setattr(module, "fit_probability_model", fail_if_refit)
    first = predict_from_rq1_snapshot(snapshot, target)
    second = predict_from_rq1_snapshot(snapshot, target)
    pd.testing.assert_frame_equal(first, second)
    assert snapshot.last_development_build_index == 200


def test_second_build_without_retraining(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    later = target.copy(deep=True)
    later["build_id"] = "b0202"
    later["build_index"] = 202
    first = predict_from_rq1_snapshot(snapshot, target)
    second = predict_from_rq1_snapshot(snapshot, later)
    assert len(first) == len(second) == 4
    assert snapshot.last_development_build_index == 200
    assert snapshot.model_fit_last_build_index < 201
    assert second["build_index"].eq(202).all()


def test_target_outcome_is_never_accepted(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    for hidden_field in ("failed", "execution_time_ms", "oracle_truth"):
        forbidden = target.copy(deep=True)
        forbidden[hidden_field] = 1
        with pytest.raises(ValueError, match="unexpected/unsafe"):
            predict_from_rq1_snapshot(snapshot, forbidden)


def test_no_target_or_future_development_leakage(cfg, synthetic_data):
    dev, target = synthetic_data
    mutable_development = dev.copy(deep=True)
    snapshot = _fit(cfg, mutable_development)
    before = predict_from_rq1_snapshot(snapshot, target)
    mutable_development["failed"] = 1 - mutable_development["failed"]
    after = predict_from_rq1_snapshot(snapshot, target)
    pd.testing.assert_frame_equal(before, after)
    assert snapshot.last_development_build_index == 200
    with pytest.raises(ValueError, match="strictly after"):
        predict_from_rq1_snapshot(snapshot, target.assign(build_index=200, build_id="b0200"))


def test_target_cross_project_rejected(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    with pytest.raises(ValueError, match="another project"):
        predict_from_rq1_snapshot(snapshot, target.assign(project_id="different-project"))


def test_target_multiple_builds_rejected(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    other = target.iloc[[0]].copy()
    other["build_id"] = "b0202"
    other["build_index"] = 202
    with pytest.raises(ValueError, match="exactly one build"):
        predict_from_rq1_snapshot(snapshot, pd.concat([target, other], ignore_index=True))


def test_target_duplicate_and_missing_feature_rejected(cfg, synthetic_data):
    dev, target = synthetic_data
    snapshot = _fit(cfg, dev)
    with pytest.raises(ValueError, match="duplicate row identity"):
        predict_from_rq1_snapshot(snapshot, pd.concat([target, target.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="missing features"):
        predict_from_rq1_snapshot(snapshot, target.drop(columns=["failure_rate_history"]))


def test_insufficient_calibration_parity(cfg, synthetic_data):
    dev, target = synthetic_data
    insufficient = dev.copy(deep=True)
    insufficient.loc[insufficient["build_index"] >= 101, "failed"] = 0
    snapshot = _fit(cfg, insufficient)
    assert not snapshot.calibration_support_sufficient, (
        "Synthetic fixture unexpectedly met frozen calibration support criteria"
    )
    assert snapshot.platt is None and snapshot.isotonic is None
    legacy, diagnostics = _legacy(cfg, insufficient, target)
    assert diagnostics.get("calibration_support_sufficient") is False
    observed = predict_from_rq1_snapshot(snapshot, target)
    _assert_prediction_parity(legacy, observed)
    assert observed[PROBABILITY_COLUMNS["platt"]].isna().all()
    assert observed[PROBABILITY_COLUMNS["isotonic"]].isna().all()


def test_bad_development_labels_rejected(cfg, synthetic_data):
    dev, _ = synthetic_data
    invalid = dev.copy(deep=True)
    invalid.loc[invalid.index[0], "failed"] = 5
    with pytest.raises(ValueError, match="binary"):
        _fit(cfg, invalid)
