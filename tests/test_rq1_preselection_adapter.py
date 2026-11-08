"""Synthetic adapter checks; monkeypatched RQ1 fit (NO real training)."""

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci import rq1_preselection_adapter as a

IDS = list(a.IDENTITY_COLUMNS)


def frames():
    development = pd.DataFrame([
        {"project_id": "p", "build_id": "b1", "build_index": 1,
         "test_id": "a", "pre_feature": 1.0, "failed": 1,
         "execution_time_ms": 999.0},
        {"project_id": "p", "build_id": "b1", "build_index": 1,
         "test_id": "b", "pre_feature": 2.0, "failed": 0,
         "execution_time_ms": 888.0},
    ])
    target = pd.DataFrame([
        {"project_id": "p", "build_id": "b2", "build_index": 2,
         "test_id": "a", "pre_feature": 3.0,
         "failed": 1, "execution_time_ms": 500.0},
        {"project_id": "p", "build_id": "b2", "build_index": 2,
         "test_id": "b", "pre_feature": 4.0,
         "failed": 0, "execution_time_ms": 600.0},
    ])
    return development, target


def install_fake(monkeypatch, *, transform=None):
    calls = []

    def validate(**kwargs):
        return ["pre_feature"]

    def fake_run(*, development, evaluation, **kwargs):
        assert "execution_time_ms" not in evaluation.columns
        assert "execution_time_ms" not in development.columns
        assert set(evaluation.failed) == {0}
        assert (development.build_index < evaluation.build_index.iloc[0]).all()
        assert "pre_feature" in evaluation.columns
        calls.append((development.copy(deep=True), evaluation.copy(deep=True)))
        output = evaluation.loc[:, IDS].copy(deep=True)
        output["failed"] = evaluation.failed
        output["p_uncalibrated"] = output.test_id.map({"a": 0.8, "b": 0.2})
        output["p_platt"] = output.test_id.map({"a": 0.7, "b": 0.3})
        output["p_isotonic"] = output.test_id.map({"a": 0.6, "b": 0.4})
        return (output if transform is None else transform(output)), {"fit_before_target": True}

    monkeypatch.setattr(a, "validate_rq1_integration_contract", validate)
    monkeypatch.setattr(a, "run_rq1_model", fake_run)
    return calls


def execute(development, target):
    return a.predict_preselection_rq1(
        development, target, "logistic_regression", {}, {}, {}, {}
    )


def test_preselection_contract_whitelist_and_dummy_target_label(monkeypatch):
    calls = install_fake(monkeypatch)
    development, target = frames()
    target_original = target.copy(deep=True)
    result, diagnostic = execute(development, target)
    assert result.columns.tolist() == IDS + list(a.PROBABILITY_COLUMNS.values())
    assert "failed" not in result.columns
    assert "execution_time_ms" not in result.columns
    assert result.p_uncalibrated.tolist() == [0.8, 0.2]
    assert diagnostic == {"fit_before_target": True}
    pd.testing.assert_frame_equal(target, target_original)
    assert len(calls) == 1


def test_target_truth_mutations_do_not_affect_predictions(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    first, _ = execute(development, target)
    target.loc[:, "failed"] = [0, 1]
    target.loc[:, "execution_time_ms"] = [90000.0, 1.0]
    second, _ = execute(development, target)
    pd.testing.assert_frame_equal(first, second)


def test_target_truth_columns_may_be_absent(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    target = target.drop(columns=["failed", "execution_time_ms"])
    result, _ = execute(development, target)
    assert len(result) == 2


def test_future_history_rejected_before_model_call(monkeypatch):
    calls = install_fake(monkeypatch)
    development, target = frames()
    development["build_index"] = 2
    with pytest.raises(ValueError, match="current/future"):
        execute(development, target)
    assert calls == []


def test_cross_project_rejected(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    target["project_id"] = "q"
    with pytest.raises(ValueError, match="same project"):
        execute(development, target)


def test_multiple_target_builds_rejected(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    target.loc[1, ["build_id", "build_index"]] = ["b3", 3]
    with pytest.raises(ValueError, match="exactly one build_index"):
        execute(development, target)


def test_duplicate_candidates_rejected(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    target = pd.concat([target, target.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        execute(development, target)


def test_output_order_is_aligned_by_identity(monkeypatch):
    install_fake(monkeypatch, transform=lambda x: x.iloc[::-1].reset_index(drop=True))
    development, target = frames()
    result, _ = execute(development, target)
    assert result.test_id.tolist() == ["a", "b"]
    assert result.p_uncalibrated.tolist() == [0.8, 0.2]


def test_missing_preexecution_feature_is_rejected(monkeypatch):
    install_fake(monkeypatch)
    development, target = frames()
    with pytest.raises(ValueError, match="missing RQ1"):
        execute(development, target.drop(columns="pre_feature"))


def test_forbidden_postexecution_feature_contract_rejected(monkeypatch):
    install_fake(monkeypatch)
    monkeypatch.setattr(a, "validate_rq1_integration_contract", lambda **kwargs: ["failed"])
    development, target = frames()
    with pytest.raises(ValueError, match="post-execution"):
        execute(development, target)


def test_partial_missing_calibration_rejected(monkeypatch):
    def corrupt(x):
        x.loc[x.test_id.eq("a"), "p_platt"] = np.nan
        return x
    install_fake(monkeypatch, transform=corrupt)
    development, target = frames()
    with pytest.raises(ValueError, match="partially missing"):
        execute(development, target)


def test_all_missing_calibration_is_preserved(monkeypatch):
    def unavailable(x):
        x["p_platt"] = np.nan
        x["p_isotonic"] = np.nan
        return x
    install_fake(monkeypatch, transform=unavailable)
    development, target = frames()
    result, _ = execute(development, target)
    assert result.p_platt.isna().all()
    assert result.p_isotonic.isna().all()


def test_invalid_uncalibrated_probability_rejected(monkeypatch):
    def corrupt(x):
        x.loc[x.test_id.eq("a"), "p_uncalibrated"] = 1.2
        return x
    install_fake(monkeypatch, transform=corrupt)
    development, target = frames()
    with pytest.raises(ValueError, match="invalid probability"):
        execute(development, target)


def test_missing_prediction_candidate_rejected(monkeypatch):
    install_fake(monkeypatch, transform=lambda x: x.iloc[[0]].copy())
    development, target = frames()
    with pytest.raises(ValueError, match="missing target candidates"):
        execute(development, target)
