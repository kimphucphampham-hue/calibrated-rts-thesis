"""Synthetic prefix-causal integration tests; no CORE, no final policy claims."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.tcp_ci.rq1_rq2_multibuild_prototype as multi
from src.tcp_ci.rq2_causal_feature_materializer import (
    materialize_target_preexecution_features as materialize,
)


def _time(idx):
    return (pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=idx - 1)).isoformat()


def _data(count=13, tests=("alpha", "beta")):
    history = pd.DataFrame([
        dict(project_id="synthetic-causal", build_id=f"b{i:04d}", build_index=i,
             build_started_at=_time(i), test_id=t,
             failed=int((i + j) % (3 + j) == 0),
             execution_time_ms=int(10 + j * 7 + i % 5),
             changed_entity_count=int(i % 5))
        for i in range(1, count + 1) for j, t in enumerate(tests)
    ])
    context = pd.DataFrame([
        dict(project_id="synthetic-causal", build_id=f"b{i:04d}", build_index=i,
             build_started_at=_time(i), test_id=t,
             changed_entity_count=int(i % 5))
        for i in (count + 1, count + 2) for t in (*tests, "gamma")
    ])
    oracle = context.loc[:, list(multi.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 1, 0, 1, 0]
    oracle["execution_time_ms"] = [25, 26, 27, 31, 32, 33]
    return history, context, oracle


def _run(history, context, oracle, **kwargs):
    return multi.run_synthetic_multibuild_rq1_rq2(
        history, context, oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=.95,
        rq1_config={}, model_config={}, calibration_config={},
        core_features_config={}, preexecution_mode="causal_from_history", **kwargs,
    )


def _spy(monkeypatch, calls):
    def stub(*, development, target_preexecution, history, target_oracle, **kwargs):
        idx = int(target_preexecution.build_index.iloc[0])
        assert development.build_index.max() < idx
        assert history.build_index.max() < idx
        assert "failed" not in target_preexecution
        assert "execution_time_ms" not in target_preexecution
        assert "execution_time_mean_history_ms" in target_preexecution
        calls.append(dict(idx=idx, development=development.copy(deep=True),
                          pre=target_preexecution.copy(deep=True),
                          history=history.copy(deep=True),
                          oracle=target_oracle.copy(deep=True)))
        return SimpleNamespace(build_index=idx, selected_test_ids=("alpha",),
                               selection_mode="stub", report={})
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", stub)


def _by_test(frame):
    return frame.sort_values("test_id").reset_index(drop=True)


def test_each_target_rematerialized_from_observed_prefix(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    result = _run(h, context, oracle)
    assert [c["idx"] for c in calls] == [14, 15]
    assert result.development_rows_before == (26, 29)
    expected14 = materialize(h, context.query("build_index == 14"))
    pd.testing.assert_frame_equal(_by_test(calls[0]["pre"]), _by_test(expected14))
    realized14 = context.query("build_index == 14").merge(
        oracle.query("build_index == 14"), on=list(multi.IDENTITY), validate="one_to_one")
    expected15 = materialize(pd.concat([h, realized14], ignore_index=True),
                             context.query("build_index == 15"))
    pd.testing.assert_frame_equal(_by_test(calls[1]["pre"]), _by_test(expected15))
    assert calls[0]["pre"].set_index("test_id").loc["gamma", "prior_execution_count"] == 0
    assert calls[1]["pre"].set_index("test_id").loc["gamma", "prior_execution_count"] == 1
    assert calls[1]["pre"].set_index("test_id").loc["gamma", "recent_failure"] == 1


def test_development_frozen_features_not_taken_from_caller(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    _run(h, context, oracle)
    before = calls[0]["development"].set_index(["build_index", "test_id"])
    assert "failure_rate_history" in before
    assert pd.isna(before.loc[(1, "alpha"), "failure_rate_history"])
    assert before.loc[(13, "alpha"), "prior_execution_count"] == 12


def test_outcome_mutation_does_not_change_same_build_features(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    _run(h, context, oracle)
    changed = oracle.copy(deep=True)
    changed.loc[changed.build_index == 14, "failed"] = [0, 1, 0]
    changed.loc[changed.build_index == 14, "execution_time_ms"] = [999, 998, 997]
    calls2 = []
    _spy(monkeypatch, calls2)
    _run(h, context, changed)
    pd.testing.assert_frame_equal(calls[0]["pre"], calls2[0]["pre"])
    assert not calls[1]["pre"].equals(calls2[1]["pre"])


def test_future_truth_changes_cannot_affect_prior_features(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    _run(h, context, oracle)
    changed = oracle.copy(deep=True)
    changed.loc[changed.build_index == 15, "failed"] = 1
    changed.loc[changed.build_index == 15, "execution_time_ms"] = 900
    calls2 = []
    _spy(monkeypatch, calls2)
    _run(h, context, changed)
    for a, b in zip(calls, calls2):
        pd.testing.assert_frame_equal(a["pre"], b["pre"])
        pd.testing.assert_frame_equal(a["development"], b["development"])


def test_prefix_equivalence_and_input_order_invariance(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    _run(h, context, oracle)
    calls2 = []
    _spy(monkeypatch, calls2)
    _run(h.sample(frac=1, random_state=4),
         context.sample(frac=1, random_state=8),
         oracle.sample(frac=1, random_state=7))
    for a, b in zip(calls, calls2):
        pd.testing.assert_frame_equal(_by_test(a["pre"]), _by_test(b["pre"]))
    calls3 = []
    _spy(monkeypatch, calls3)
    _run(h, context.query("build_index == 14"), oracle.query("build_index == 14"))
    pd.testing.assert_frame_equal(calls[0]["pre"], calls3[0]["pre"])


def test_target_is_preexecution_only_and_f2_retained(monkeypatch):
    h, context, oracle = _data()
    calls = []
    _spy(monkeypatch, calls)
    _run(h, context, oracle)
    assert set(calls[0]["pre"]["changed_entity_count"]) == {14 % 5}
    assert set(calls[1]["pre"]["changed_entity_count"]) == {15 % 5}
    assert "failed" not in calls[1]["pre"]
    assert "execution_time_ms" not in calls[1]["pre"]


@pytest.mark.parametrize("change, message", [
    ("target_bad_feature", "exactly pre-execution context"),
    ("target_outcome", "exactly pre-execution context"),
    ("development_bad_feature", "exactly raw canonical"),
    ("history_future", "overlaps"),
    ("context_earlier_timestamp", "Historical build_started_at follows"),
    ("negative_change_count", "invalid changed_entity_count"),
])
def test_causal_input_fail_closed(monkeypatch, change, message):
    h, context, oracle = _data()
    if change == "target_bad_feature":
        context["failure_rate_history"] = 0.0
    elif change == "target_outcome":
        context["failed"] = 0
    elif change == "development_bad_feature":
        h["recent_failure"] = 0
    elif change == "history_future":
        h.loc[h.build_index == 13, "build_index"] = 14
    elif change == "context_earlier_timestamp":
        context.loc[context.build_index == 14, "build_started_at"] = _time(4)
    elif change == "negative_change_count":
        context.loc[context.build_index == 14, "changed_entity_count"] = -5
    _spy(monkeypatch, [])
    with pytest.raises(ValueError, match=message):
        _run(h, context, oracle)


def test_unsupported_mode_fails_closed():
    h, context, oracle = _data()
    with pytest.raises(ValueError, match="Unsupported preexecution_mode"):
        multi.run_synthetic_multibuild_rq1_rq2(
            h, context, oracle, model_name="logistic_regression",
            probability_variant="uncalibrated", rho=.95,
            rq1_config={}, model_config={}, calibration_config={},
            core_features_config={}, preexecution_mode="unknown",
        )


def test_real_rq1_two_builds_with_causal_features():
    """Bounded no-mock smoke, using frozen F1/F2/F3; NOT an RQ3 schedule."""
    from src.tcp_ci.rq1_pipeline import load_yaml
    cfg = dict(
        rq1_config=load_yaml(Path("configs/rq1_v1.yaml")),
        model_config=load_yaml(Path("configs/rq1_models_v1.yaml")),
        calibration_config=load_yaml(Path("configs/calibration_v1.yaml")),
        core_features_config=load_yaml(Path("configs/core_features_v1.yaml")),
    )
    tests = ("alpha", "beta", "gamma", "delta")
    raw = pd.DataFrame([
        dict(project_id="synthetic-causal-real", build_id=f"b{i:04d}",
             build_index=i, build_started_at=_time(i), test_id=t,
             failed=int((j == 0 and i % 3 == 0) or (j == 1 and i % 5 == 0)
                        or (j == 2 and i % 11 == 0)),
             execution_time_ms=int(9 + 8 * j + i % 3),
             changed_entity_count=int(i % 5))
        for i in range(1, 201) for j, t in enumerate(tests)
    ])
    context = pd.DataFrame([
        dict(project_id="synthetic-causal-real", build_id=f"b{i:04d}",
             build_index=i, build_started_at=_time(i), test_id=t,
             changed_entity_count=int(i % 5))
        for i in (201, 202) for t in tests
    ])
    oracle = context.loc[:, list(multi.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 0, 0, 0, 1, 0, 0]
    oracle["execution_time_ms"] = [12, 18, 24, 30, 14, 20, 26, 32]
    result = multi.run_synthetic_multibuild_rq1_rq2(
        raw, context, oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=.95,
        preexecution_mode="causal_from_history", **cfg,
    )
    assert [step.build_index for step in result.steps] == [201, 202]
    assert result.development_rows_before == (800, 804)
    for step in result.steps:
        assert len(step.prediction_view) == 4
        assert step.selection_mode in {"normal_risk_density", "fallback_retest_all"}
        assert step.report.get("observation_regime") == "retrospective_full_information"
        assert set(step.selected_test_ids).issubset(set(tests))
