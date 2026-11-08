"""Synthetic multi-build integration invariants (no CORE data).

Mocked single-build bridge tests orchestration state. One bounded real RQ1
smoke test validates the end-to-end helper on a tiny synthetic chronology.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.tcp_ci.rq1_rq2_multibuild_prototype as multi


def _tiny():
    tests = ("alpha", "beta")
    seed = pd.DataFrame([
        {"project_id": "synthetic", "build_id": f"b{i}", "build_index": i,
         "test_id": t, "failed": int(i == 3 and t == "alpha"),
         "execution_time_ms": float(8 + i + j),
         "execution_time_mean_history_ms": float(10 + j)}
        for i in range(1, 4) for j, t in enumerate(tests)
    ])
    pre = pd.DataFrame([
        {"project_id": "synthetic", "build_id": f"b{i}", "build_index": i,
         "test_id": t, "execution_time_mean_history_ms": float(10 + j)}
        for i in range(4, 7) for j, t in enumerate(tests)
    ])
    oracle = pre.loc[:, list(multi.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 0, 1, 0, 0]
    oracle["execution_time_ms"] = [17.0, 18.0, 20.0, 21.0, 18.0, 22.0]
    return seed, pre, oracle


def _run(seed, pre, oracle):
    return multi.run_synthetic_multibuild_rq1_rq2(
        seed, pre, oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=0.95,
        rq1_config={}, model_config={}, calibration_config={},
        core_features_config={},
    )


def _spy(monkeypatch, calls):
    def stub(*, development, target_preexecution, history, target_oracle,
             **kwargs):
        idx = int(target_preexecution["build_index"].iloc[0])
        assert development["build_index"].max() < idx
        assert history["build_index"].max() < idx
        assert "failed" not in target_preexecution.columns
        assert "execution_time_ms" not in target_preexecution.columns
        calls.append((idx, development.copy(), history.copy(), target_oracle.copy()))
        selected = ("alpha",) if int(development["failed"].sum()) % 2 else ("beta",)
        return SimpleNamespace(build_index=idx, selected_test_ids=selected,
                               selection_mode="stub", report={})
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", stub)


def test_chronological_loop_and_full_feedback(monkeypatch):
    seed, pre, oracle = _tiny()
    calls = []
    _spy(monkeypatch, calls)
    result = _run(seed, pre.sample(frac=1, random_state=3), oracle.sample(frac=1, random_state=4))
    assert [s.build_index for s in result.steps] == [4, 5, 6]
    assert result.development_rows_before == (6, 8, 10)
    assert result.history_rows_before == (6, 8, 10)
    assert sorted(set(calls[1][1]["build_index"])) == [1, 2, 3, 4]
    known4 = calls[1][1].query("build_index == 4").set_index("test_id")
    assert known4.loc["alpha", "failed"] == 1
    assert known4.loc["beta", "execution_time_ms"] == 18
    assert result.observation_regime == "retrospective_full_information"


def test_prefix_equivalence(monkeypatch):
    seed, pre, oracle = _tiny()
    _spy(monkeypatch, [])
    full = _run(seed, pre, oracle)
    prefix = _run(seed, pre.query("build_index <= 5"), oracle.query("build_index <= 5"))
    assert [(s.build_index, s.selected_test_ids) for s in full.steps[:2]] == [
        (s.build_index, s.selected_test_ids) for s in prefix.steps]


def test_future_oracle_mutation_does_not_change_earlier_decisions(monkeypatch):
    seed, pre, oracle = _tiny()
    _spy(monkeypatch, [])
    before = _run(seed, pre, oracle)
    other = oracle.copy()
    other.loc[other.build_index == 6, "failed"] = 1
    other.loc[other.build_index == 6, "execution_time_ms"] = 500
    after = _run(seed, pre, other)
    assert [s.selected_test_ids for s in before.steps] == [s.selected_test_ids for s in after.steps]


def test_past_oracle_updates_later_history_but_not_current_decision(monkeypatch):
    seed, pre, oracle = _tiny()
    _spy(monkeypatch, [])
    baseline = _run(seed, pre, oracle)
    changed = oracle.copy()
    changed.loc[changed.build_index == 4, "failed"] = 0
    alternate = _run(seed, pre, changed)
    assert baseline.steps[0].selected_test_ids == alternate.steps[0].selected_test_ids
    assert baseline.steps[1].selected_test_ids != alternate.steps[1].selected_test_ids


def test_input_frames_not_mutated(monkeypatch):
    seed, pre, oracle = _tiny()
    originals = [frame.copy(deep=True) for frame in (seed, pre, oracle)]
    _spy(monkeypatch, [])
    _run(seed, pre, oracle)
    for actual, expected in zip((seed, pre, oracle), originals):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("change,match", [
    ("target_outcome_in_pre", "post-execution"),
    ("overlapping_seed", "overlaps"),
    ("missing_oracle", "identities differ"),
    ("duplicate_candidate", "duplicate"),
    ("cross_project", "projects differ"),
    ("invalid_build_index", "invalid build_index"),
    ("invalid_oracle_duration", "invalid realized durations"),
    ("invalid_oracle_label", "invalid failure labels"),
    ("oracle_extra_field", "Oracle must contain only"),
])
def test_bad_input_fails_closed(monkeypatch, change, match):
    seed, pre, oracle = _tiny()
    if change == "target_outcome_in_pre":
        pre["failed"] = 0
    elif change == "overlapping_seed":
        seed.loc[seed.index[0], ["build_index", "build_id"]] = [4, "b4"]
    elif change == "missing_oracle":
        oracle = oracle.iloc[:-1].copy()
    elif change == "duplicate_candidate":
        pre = pd.concat([pre, pre.iloc[[0]]], ignore_index=True)
    elif change == "cross_project":
        oracle["project_id"] = "other"
    elif change == "invalid_build_index":
        pre = pre.astype({"build_index": float})
        pre.loc[pre.index[0], "build_index"] = 4.5
    elif change == "invalid_oracle_duration":
        oracle.loc[oracle.index[0], "execution_time_ms"] = -1
    elif change == "invalid_oracle_label":
        oracle.loc[oracle.index[0], "failed"] = 4
    elif change == "oracle_extra_field":
        oracle["future_outcome"] = 123
    _spy(monkeypatch, [])
    with pytest.raises(ValueError, match=match):
        _run(seed, pre, oracle)


def test_real_rq1_two_build_smoke():
    """Actual RQ1 LR + calibrators + RQ2: no fit monkeypatch.

    This synthetic fixture provides predetermined features, not validated
    materialized features from TCP-CI. Per-build refit is NOT RQ3 policy.
    """
    from src.tcp_ci.rq1_pipeline import (
        load_yaml, validate_rq1_integration_contract,
    )
    cfg = dict(
        rq1_config=load_yaml(Path("configs/rq1_v1.yaml")),
        model_config=load_yaml(Path("configs/rq1_models_v1.yaml")),
        calibration_config=load_yaml(Path("configs/calibration_v1.yaml")),
        core_features_config=load_yaml(Path("configs/core_features_v1.yaml")),
    )
    features = list(validate_rq1_integration_contract(**cfg))
    assert "build_index" in features
    tests = ("alpha", "beta", "gamma", "delta")

    def feature_values(idx, j):
        out = {}
        for pos, field in enumerate(features):
            k = field.lower()
            if k == "build_index":
                value = int(idx)
            elif k == "failure_rate_history":
                value = float((idx + j) % 10) / 10
            elif k == "recent_failure":
                value = int((idx + j) % 5 == 0)
            elif k == "execution_time_mean_history_ms":
                value = float(10 + 8 * j)
            elif k == "test_age_builds":
                value = idx - 1
            elif "since_last_failure" in k:
                value = (idx + j) % 12
            elif "consecutive_failures" in k:
                value = (idx + j) % 3
            elif "consecutive_passes" in k:
                value = (idx + 2 * j) % 4
            elif "changed_entity" in k:
                value = (idx + j) % 5
            else:
                value = float((idx * (pos + 3) + j * 7) % 23) / 11
            out[field] = value
        return out

    def row(idx, j, test):
        return {"project_id": "synthetic-multibuild-rq1", "build_id": f"b{idx:04d}",
                "build_index": idx, "test_id": test, **feature_values(idx, j)}

    initial = pd.DataFrame([
        dict(row(idx, j, test), failed=int((j == 0 and idx % 3 == 0) or
                                         (j == 1 and idx % 5 == 0) or
                                         (j == 2 and idx % 11 == 0)),
             execution_time_ms=float(9 + 8 * j + idx % 3))
        for idx in range(1, 201) for j, test in enumerate(tests)
    ])
    pre = pd.DataFrame([row(idx, j, test) for idx in (201, 202) for j, test in enumerate(tests)])
    oracle = pre.loc[:, list(multi.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 0, 0, 0, 1, 0, 0]
    oracle["execution_time_ms"] = [12., 18., 24., 30., 14., 20., 26., 32.]
    result = multi.run_synthetic_multibuild_rq1_rq2(
        initial, pre, oracle,
        model_name="logistic_regression", probability_variant="uncalibrated",
        rho=.95, **cfg,
    )
    assert [x.build_index for x in result.steps] == [201, 202]
    assert result.development_rows_before == (800, 804)
    for step in result.steps:
        assert len(step.prediction_view) == 4
        assert step.selection_mode in {"normal_risk_density", "fallback_retest_all"}
        assert step.report.get("observation_regime") == "retrospective_full_information"
        assert step.rq1_diagnostics["calibration_support_sufficient"] is True
        assert set(step.selected_test_ids).issubset(set(tests))
