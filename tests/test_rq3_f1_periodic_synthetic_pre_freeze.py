"""Bounded pre-freeze RQ3 F1 static/periodic prototype tests; NOT CORE evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.tcp_ci.rq1_rq2_multibuild_prototype as multi
from src.tcp_ci.rq1_pipeline import load_yaml


def _utc(i):
    return (pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=i)).isoformat()


def _raw(initial_builds=13, n_targets=26):
    tests = ("alpha", "beta", "gamma", "delta")
    def row(i, t, j):
        return dict(project_id="synthetic-rq3-f1", build_id=f"b{i:05d}",
                    build_index=i, test_id=t, build_started_at=_utc(i),
                    failed=int((i + j) % (4 + j) == 0),
                    execution_time_ms=float(11 + 7*j + i % 3))
    history = pd.DataFrame([row(i, t, j) for i in range(1, initial_builds + 1)
                            for j, t in enumerate(tests)])
    future = pd.DataFrame([row(i, t, j) for i in range(initial_builds + 1, initial_builds+n_targets+1)
                           for j, t in enumerate(tests)])
    target = future.loc[:, list(multi.IDENTITY) + ["build_started_at"]].copy()
    oracle = future.loc[:, list(multi.ORACLE_FIELDS)].copy()
    return history, target, oracle


def _cfg():
    base = Path("configs")
    config = dict(rq1_config=load_yaml(base / "rq1_v1.yaml"),
                  model_config=load_yaml(base / "rq1_models_v1.yaml"),
                  calibration_config=load_yaml(base / "calibration_v1.yaml"),
                  core_features_config=load_yaml(base / "core_features_v1.yaml"))
    config["rq1_config"]["feature_set"]["name"] = "F1"
    config["rq1_config"]["pre_freeze_feature_scope"] = "F1_ONLY_V1"
    config["model_config"]["feature_config"]["primary_feature_set"] = "F1"
    return config


def _run(history, pre, oracle, *, mode, h=None, cfg=None):
    return multi.run_synthetic_multibuild_rq1_rq2(
        history, pre, oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=.95,
        snapshot_mode=mode, periodic_h=h, preexecution_mode="causal_from_history",
        **(cfg if cfg is not None else _cfg()),
    )


def _spy(monkeypatch):
    fits, decisions = [], []
    def fit(prefix, **kwargs):
        assert "changed_entity_count" in prefix.columns  # private F2 sentinel
        assert prefix.changed_entity_count.isna().all()
        last = int(prefix.build_index.max())
        fits.append(last)
        return SimpleNamespace(last_development_build_index=last)
    def bridge(*, development, target_preexecution, history, target_oracle, rq1_snapshot=None, **kwargs):
        idx = int(target_preexecution.build_index.iloc[0])
        assert rq1_snapshot is not None
        assert rq1_snapshot.last_development_build_index < idx
        assert int(history.build_index.max()) < idx
        assert int(development.build_index.max()) < idx
        assert "failed" not in target_preexecution
        assert "execution_time_ms" not in target_preexecution
        assert "changed_entity_count" not in target_preexecution
        decisions.append((idx, rq1_snapshot.last_development_build_index,
                          target_preexecution.copy(deep=True)))
        return SimpleNamespace(build_index=idx, selected_test_ids=("alpha",),
                               selection_mode="stub", report={})
    monkeypatch.setattr(multi, "fit_rq1_snapshot", fit)
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", bridge)
    return fits, decisions


@pytest.mark.parametrize("h", [5, 10, 20])
def test_periodic_retrains_at_exact_target_positions_from_strict_prefix(monkeypatch, h):
    history, target, oracle = _raw()
    fits, decisions = _spy(monkeypatch)
    output = _run(history, target, oracle, mode="periodic_synthetic", h=h)
    targets = list(range(14, 40))
    expected_targets = targets[::h]
    assert output.snapshot_fit_at_target_indices == tuple(expected_targets)
    assert fits == [idx - 1 for idx in expected_targets]
    assert len(decisions) == len(targets)
    assert output.snapshot_last_development_by_target == tuple(cutoff for _,cutoff,_ in decisions)
    assert list(output.development_rows_before) == [len(history) + 4*i for i in range(26)]


def test_static_fits_only_once_and_reuses_same_snapshot(monkeypatch):
    history, target, oracle = _raw(n_targets=21)
    fits, decisions = _spy(monkeypatch)
    result = _run(history, target, oracle, mode="fixed_initial_synthetic")
    assert fits == [13]
    assert result.snapshot_fit_at_target_indices == (14,)
    assert result.snapshot_last_development_by_target == (13,) * 21
    assert [cutoff for _, cutoff, _ in decisions] == [13]*21


@pytest.mark.parametrize("h", [None, 0, 1, 4, 6, 30, -5, True, 5.0, "5"])
def test_periodic_policy_bad_h_fails_closed(h):
    history,target,oracle = _raw(n_targets=1)
    with pytest.raises(ValueError, match="periodic_h"):
        _run(history,target,oracle,mode="periodic_synthetic",h=h)


def test_reject_h_without_periodic_and_require_f1_causal():
    history,target,oracle=_raw(n_targets=1)
    with pytest.raises(ValueError, match="periodic_h requires"):
        _run(history,target,oracle,mode="fixed_initial_synthetic",h=5)
    cfg=_cfg()
    del cfg["rq1_config"]["pre_freeze_feature_scope"]
    with pytest.raises(ValueError, match="causal F1_ONLY_V1"):
        _run(history,target,oracle,mode="periodic_synthetic",h=5,cfg=cfg)
    with pytest.raises(ValueError, match="requires causal_from_history"):
        multi.run_synthetic_multibuild_rq1_rq2(
            history,target,oracle,model_name="logistic_regression",
            probability_variant="uncalibrated",rho=.95,
            preexecution_mode="supplied_synthetic",
            snapshot_mode="fixed_initial_synthetic",**_cfg())


def test_f1_raw_rejects_f2_target_oracle_and_future_training(monkeypatch):
    history,target,oracle=_raw(n_targets=2)
    _spy(monkeypatch)
    _run(history,target,oracle,mode="fixed_initial_synthetic")
    with pytest.raises(ValueError, match="exactly pre-execution"):
        _run(history,target.assign(changed_entity_count=0),oracle,mode="fixed_initial_synthetic")
    with pytest.raises(ValueError, match="exactly pre-execution"):
        _run(history,target.assign(failed=1),oracle,mode="fixed_initial_synthetic")
    with pytest.raises(ValueError, match="exactly raw canonical"):
        _run(history.assign(changed_entity_count=0),target,oracle,mode="fixed_initial_synthetic")
    with pytest.raises(ValueError, match="overlaps"):
        _run(pd.concat([history,target.assign(failed=0,execution_time_ms=1)],ignore_index=True),
             target,oracle,mode="fixed_initial_synthetic")


def test_last_target_oracle_mutation_cannot_change_any_prior_prediction_inputs(monkeypatch):
    history,target,oracle=_raw(n_targets=7)
    _, calls1 = _spy(monkeypatch)
    _run(history,target,oracle,mode="periodic_synthetic",h=5)
    changed=oracle.copy(deep=True)
    changed.loc[changed.build_index == 20,"failed"] = 1
    changed.loc[changed.build_index == 20,"execution_time_ms"] = 999
    _, calls2=_spy(monkeypatch)
    _run(history,target,changed,mode="periodic_synthetic",h=5)
    assert [(i,c) for i,c,_ in calls1] == [(i,c) for i,c,_ in calls2]
    for (_,_,a),(_,_,b) in zip(calls1,calls2):
        pd.testing.assert_frame_equal(a,b)


def test_real_lr_f1_static_and_periodic_end_to_end_synthetic():
    history,target,oracle=_raw(initial_builds=200,n_targets=6)
    static=_run(history,target,oracle,mode="fixed_initial_synthetic")
    periodic=_run(history,target,oracle,mode="periodic_synthetic",h=5)
    assert static.snapshot_fit_at_target_indices == (201,)
    assert periodic.snapshot_fit_at_target_indices == (201,206)
    assert periodic.snapshot_last_development_by_target == (200,200,200,200,200,205)
    for result in (static,periodic):
        assert len(result.steps)==6
        for step in result.steps:
            assert len(step.prediction_view)==4
            assert step.prediction_view["p_uncalibrated"].between(0,1).all()
            assert step.rq1_diagnostics.get("calibration_support_sufficient") is True
            assert step.report.get("observation_regime") == "retrospective_full_information"
    oracle2=oracle.copy(deep=True)
    oracle2.loc[oracle2.build_index==206,"failed"] = 1
    oracle2.loc[oracle2.build_index==206,"execution_time_ms"] = 999
    periodic2=_run(history,target,oracle2,mode="periodic_synthetic",h=5)
    for a,b in zip(periodic.steps,periodic2.steps):
        pd.testing.assert_frame_equal(a.prediction_view,b.prediction_view)
        assert a.selected_test_ids == b.selected_test_ids
