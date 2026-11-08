"""Synthetic tests for RQ1 adapter -> RQ2 REAL selection/scoring bridge.

RQ1 fitting remains stubbed; RQ2 modules are imported unchanged from repo.
"""
import numpy as np
import pandas as pd
import pytest
from src.tcp_ci import rq1_rq2_single_build_bridge as bridge


def fixture_frames():
    development = pd.DataFrame([
        dict(project_id="p", build_id="b1", build_index=1, test_id="a", failed=1, f=1),
        dict(project_id="p", build_id="b1", build_index=1, test_id="b", failed=0, f=2),
    ])
    target = pd.DataFrame([
        dict(project_id="p", build_id="b2", build_index=2, test_id="a", f=3, execution_time_mean_history_ms=10),
        dict(project_id="p", build_id="b2", build_index=2, test_id="b", f=4, execution_time_mean_history_ms=20),
    ])
    oracle = target.loc[:, ["project_id", "build_id", "build_index", "test_id"]].copy()
    oracle["failed"] = [1, 0]
    oracle["execution_time_ms"] = [100, 200]
    history = development.loc[:, ["project_id", "build_index", "test_id"]].copy()
    history["execution_time_ms"] = [10, 20]
    return development, target, history, oracle


def install_fake(monkeypatch, *, probabilities=None, platt_unavailable=False):
    calls = []
    def predict(*, development, target_preexecution, **kwargs):
        assert "failed" not in target_preexecution
        assert "execution_time_ms" not in target_preexecution
        assert (development.build_index < target_preexecution.build_index.iloc[0]).all()
        calls.append((development.copy(deep=True), target_preexecution.copy(deep=True)))
        result = target_preexecution.loc[:, list(bridge.IDENTITY)].copy(deep=True)
        values = probabilities or {"a": 0.8, "b": 0.2}
        result["p_uncalibrated"] = result.test_id.map(values)
        result["p_platt"] = np.nan if platt_unavailable else result["p_uncalibrated"] * 0.95
        result["p_isotonic"] = result["p_uncalibrated"] * 0.90
        return result.iloc[::-1].reset_index(drop=True), {"synthetic": True}
    monkeypatch.setattr(bridge, "predict_preselection_rq1", predict)
    return calls


def execute(frames, variant="uncalibrated"):
    return bridge.evaluate_one_build_with_rq1(
        *frames, model_name="logistic_regression", probability_variant=variant,
        rho=0.5, rq1_config={}, model_config={}, calibration_config={},
        core_features_config={},
    )


def test_selection_reporting_bridge_and_id_alignment(monkeypatch):
    seen = install_fake(monkeypatch)
    result = execute(fixture_frames())
    assert result.selected_test_ids == ("a",)
    assert result.selection_mode == "normal_risk_density"
    assert result.report["observed_selected_ms"] == 100.0
    assert result.probability_variant == "uncalibrated"
    assert "failed" not in result.prediction_view
    assert len(seen) == 1


def test_target_truth_mutation_changes_scoring_only(monkeypatch):
    install_fake(monkeypatch)
    original = list(fixture_frames())
    changed = list(fixture_frames())
    changed[-1].loc[changed[-1].test_id.eq("a"), "failed"] = 0
    changed[-1].loc[changed[-1].test_id.eq("a"), "execution_time_ms"] = 999
    a = execute(original)
    b = execute(changed)
    assert a.selected_test_ids == b.selected_test_ids
    assert a.report["observed_selected_ms"] != b.report["observed_selected_ms"]
    pd.testing.assert_frame_equal(a.prediction_view, b.prediction_view)


def test_preexecution_target_truth_columns_rejected(monkeypatch):
    seen = install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[1]["failed"] = [1, 0]
    with pytest.raises(ValueError, match="post-execution"):
        execute(frames)
    assert seen == []


def test_score_oracle_id_mismatch_rejected(monkeypatch):
    install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[-1].loc[0, "test_id"] = "X"
    with pytest.raises(ValueError, match="identities do not match"):
        execute(frames)


def test_history_future_results_rejected(monkeypatch):
    seen = install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[2].loc[0, "build_index"] = 2
    with pytest.raises(ValueError, match="target/future"):
        execute(frames)
    assert seen == []


def test_history_cross_project_rejected(monkeypatch):
    seen = install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[2].loc[0, "project_id"] = "q"
    with pytest.raises(ValueError, match="only the target project"):
        execute(frames)
    assert seen == []


def test_unavailable_calibration_variant_not_silently_used(monkeypatch):
    seen = install_fake(monkeypatch, platt_unavailable=True)
    with pytest.raises(ValueError, match="unavailable or invalid"):
        execute(fixture_frames(), variant="platt")
    assert len(seen) == 1


def test_unsupported_variant_rejected_before_model(monkeypatch):
    seen = install_fake(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported RQ1 probability variant"):
        execute(fixture_frames(), variant="nonsense")
    assert seen == []


def test_target_candidate_duplicates_rejected(monkeypatch):
    install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[1] = pd.concat([frames[1], frames[1].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate candidate"):
        execute(frames)


def test_target_candidate_missing_cost_feature_rejected(monkeypatch):
    install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[1] = frames[1].drop(columns="execution_time_mean_history_ms")
    with pytest.raises(ValueError, match="execution_time_mean_history_ms"):
        execute(frames)


def test_no_eligible_positive_history_uses_retest_all(monkeypatch):
    install_fake(monkeypatch)
    frames = list(fixture_frames())
    frames[2]["execution_time_ms"] = 0
    frames[1]["execution_time_mean_history_ms"] = np.nan
    result = execute(frames)
    assert result.selection_mode == "fallback_retest_all"
    assert set(result.selected_test_ids) == {"a", "b"}


def test_zero_predicted_risk_is_not_silently_approved(monkeypatch):
    install_fake(monkeypatch, probabilities={"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError):
        execute(fixture_frames())


def test_probabilities_variant_can_be_changed_without_accessing_oracle(monkeypatch):
    install_fake(monkeypatch)
    result = execute(fixture_frames(), variant="isotonic")
    assert result.probability_variant == "isotonic"
    assert result.selected_test_ids == ("a",)


def test_input_frames_are_not_modified(monkeypatch):
    install_fake(monkeypatch)
    frames = fixture_frames()
    prior = [x.copy(deep=True) for x in frames]
    execute(frames)
    for before, after in zip(prior, frames):
        pd.testing.assert_frame_equal(before, after)
