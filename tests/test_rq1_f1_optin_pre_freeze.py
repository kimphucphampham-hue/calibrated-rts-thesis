"""Pre-freeze F1-only opt-in guards; synthetic inputs, NOT final CORE evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci.build_history_features import compute_history_features
from src.tcp_ci.rq1_pipeline import (
    F1_ONLY_FEATURES,
    load_yaml,
    validate_rq1_integration_contract,
)
from src.tcp_ci.rq1_preselection_adapter import predict_preselection_rq1
from src.tcp_ci.rq2_causal_feature_materializer import (
    F1_ONLY_DECISION_COLUMNS,
    materialize_target_preexecution_features,
)


def _configs():
    d = Path("configs")
    return {
        "rq1_config": load_yaml(d / "rq1_v1.yaml"),
        "model_config": load_yaml(d / "rq1_models_v1.yaml"),
        "calibration_config": load_yaml(d / "calibration_v1.yaml"),
        "core_features_config": load_yaml(d / "core_features_v1.yaml"),
    }


def _f1_configs():
    configs = deepcopy(_configs())
    configs["rq1_config"]["feature_set"]["name"] = "F1"
    configs["rq1_config"]["pre_freeze_feature_scope"] = "F1_ONLY_V1"
    configs["model_config"]["feature_config"]["primary_feature_set"] = "F1"
    return configs


def _time(index):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat()


def _history(n=3):
    tests = ("a", "b", "c", "d")
    return pd.DataFrame([
        {
            "project_id": "synthetic-f1", "build_id": f"b{i:04d}",
            "build_index": i, "build_started_at": _time(i),
            "test_id": test,
            "failed": int((j == 0 and i % 3 == 0)
                          or (j == 1 and i % 5 == 0)
                          or (j == 2 and i % 11 == 0)),
            "execution_time_ms": 10 + j * 8 + i % 3,
        }
        for i in range(1, n + 1)
        for j, test in enumerate(tests)
    ])


def _target(index=4):
    return pd.DataFrame([
        {"project_id": "synthetic-f1", "build_id": f"b{index:04d}",
         "build_index": index, "build_started_at": _time(index),
         "test_id": test}
        for test in ("a", "b", "c", "d")
    ])


def test_legacy_11_contract_is_unchanged():
    configs = _configs()
    assert len(validate_rq1_integration_contract(**configs)) == 11
    assert configs["core_features_config"]["primary_feature_set"] == "F1_F2_F3"


def test_f1_only_acceptance_requires_explicit_opt_in():
    configs = _f1_configs()
    assert tuple(validate_rq1_integration_contract(**configs)) == F1_ONLY_FEATURES
    assert configs["core_features_config"]["primary_feature_set"] == "F1_F2_F3"
    without_optin = deepcopy(configs)
    del without_optin["rq1_config"]["pre_freeze_feature_scope"]
    with pytest.raises(ValueError, match="frozen CORE"):
        validate_rq1_integration_contract(**without_optin)


@pytest.mark.parametrize("mutation, expected", [
    ("unknown_scope", "Unsupported pre-freeze"),
    ("wrong_feature_set", "requires the F1"),
    ("changed_core_primary", "preserve frozen CORE"),
    ("altered_f1_features", "exactly seven"),
    ("claims_final", "cannot claim primary final"),
])
def test_f1_scope_fail_closed_on_method_drift(mutation, expected):
    configs = _f1_configs()
    if mutation == "unknown_scope":
        configs["rq1_config"]["pre_freeze_feature_scope"] = "F1_F2"
    elif mutation == "wrong_feature_set":
        configs["rq1_config"]["feature_set"]["name"] = "F1_F2_F3"
        configs["model_config"]["feature_config"]["primary_feature_set"] = "F1_F2_F3"
    elif mutation == "changed_core_primary":
        configs["core_features_config"]["primary_feature_set"] = "F1"
    elif mutation == "altered_f1_features":
        configs["core_features_config"]["feature_sets"]["F1"]["features"][-1] = "build_index"
    else:
        configs["rq1_config"]["scope"]["primary_final_evidence"] = True
    with pytest.raises(ValueError, match=expected):
        validate_rq1_integration_contract(**configs)


def test_f1_materializer_does_not_require_f2_and_reuses_existing_f1():
    history, context = _history(), _target()
    original = context.copy(deep=True)
    out = materialize_target_preexecution_features(
        history, context, feature_scope="F1_ONLY_V1"
    )
    assert tuple(out.columns) == F1_ONLY_DECISION_COLUMNS
    assert set(F1_ONLY_FEATURES).issubset(out.columns)
    assert "changed_entity_count" not in out.columns
    assert "recent_failure_density_10" not in out.columns
    assert "failed" not in out.columns
    assert "execution_time_ms" not in out.columns
    pd.testing.assert_frame_equal(context, original)
    expected = compute_history_features(pd.concat([
        history.assign(changed_entity_count=np.nan),
        context.assign(changed_entity_count=np.nan, failed=0, execution_time_ms=0),
    ], ignore_index=True))
    expected = expected.query("build_index == 4").sort_values("test_id").reset_index(drop=True)
    for column in F1_ONLY_FEATURES:
        pd.testing.assert_series_equal(out[column], expected[column], check_names=False)


def test_f1_materializer_is_prefix_only_and_rejects_target_oracle():
    history, target = _history(), _target()
    baseline = materialize_target_preexecution_features(
        history, target, feature_scope="F1_ONLY_V1"
    )
    for field in ("failed", "execution_time_ms", "changed_entity_count", "failure_rate_history"):
        with pytest.raises(ValueError, match="unexpected"):
            materialize_target_preexecution_features(
                history, target.assign(**{field: 123}), feature_scope="F1_ONLY_V1"
            )
    with pytest.raises(ValueError, match="strictly earlier"):
        materialize_target_preexecution_features(
            pd.concat([history, _history(4).query("build_index == 4")]),
            target, feature_scope="F1_ONLY_V1"
        )
    with pytest.raises(ValueError, match="projects differ"):
        materialize_target_preexecution_features(
            history, target.assign(project_id="other"), feature_scope="F1_ONLY_V1"
        )
    with pytest.raises(ValueError, match="Unsupported feature_scope"):
        materialize_target_preexecution_features(
            history, target, feature_scope="F1_F2"
        )
    changed_future = _history(5)
    changed_future.loc[changed_future.build_index == 5, "failed"] = 1
    retained = changed_future[changed_future.build_index < 4]
    second = materialize_target_preexecution_features(
        retained, target, feature_scope="F1_ONLY_V1"
    )
    pd.testing.assert_frame_equal(baseline, second)
    with pytest.raises(ValueError, match="changed_entity_count"):
        materialize_target_preexecution_features(history, target)  # legacy default


def test_real_logistic_f1_preselection_synthetic_no_oracle():
    """Real LR + calibration on synthetic 200-build prefix; NOT final CORE."""
    history = _history(200)
    target = _target(201)
    materialized = compute_history_features(history)
    preselection = materialize_target_preexecution_features(
        history, target, feature_scope="F1_ONLY_V1"
    )
    development = materialized.loc[:, [
        "project_id", "build_id", "build_index", "test_id", "failed", *F1_ONLY_FEATURES
    ]].copy()
    configs = _f1_configs()
    first, diagnostics = predict_preselection_rq1(
        development, preselection, "logistic_regression", **configs
    )
    assert len(first) == len(target)
    assert set(first.columns) == {
        "project_id", "build_id", "build_index", "test_id",
        "p_uncalibrated", "p_platt", "p_isotonic",
    }
    assert first["p_uncalibrated"].between(0, 1).all()
    assert diagnostics.get("calibration_support_sufficient") is True
    target_truth = preselection.copy(deep=True)
    target_truth["failed"] = [0, 1, 0, 1]
    target_truth["execution_time_ms"] = [100, 200, 300, 400]
    second, _ = predict_preselection_rq1(
        development, target_truth, "logistic_regression", **configs
    )
    pd.testing.assert_frame_equal(first, second)
