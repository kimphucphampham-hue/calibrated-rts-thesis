"""Synthetic-only causal H2 opt-in wiring tests; no CORE or Method Freeze."""

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci import rq1_rq2_single_build_bridge as bridge
from src.tcp_ci import rq1_rq2_multibuild_prototype as multi


def _single_frames():
    project = "synthetic-h2"
    history = pd.DataFrame([
        dict(project_id=project, build_index=i, test_id=t,
             execution_time_ms=c)
        for i in (1, 2) for t, c in (("a", 10.0), ("b", 5.0))
    ])
    target = pd.DataFrame([
        dict(project_id=project, build_id="build-3", build_index=3,
             test_id=t, execution_time_mean_history_ms=c,
             prior_execution_count=n, failure_rate_history=r)
        for t, c, n, r in (("a", 10.0, 2, .5),
                           ("b", 5.0, 2, 0.0),
                           ("c", np.nan, 0, np.nan))
    ])
    oracle = target.loc[:, list(bridge.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 0]
    oracle["execution_time_ms"] = [13, 7, 17]
    return history, target, oracle


def _mock_predict(monkeypatch, probabilities=(.9, .06, .04)):
    def fake_predict(*, development, target_preexecution, **kwargs):
        assert "failed" not in target_preexecution
        assert "execution_time_ms" not in target_preexecution
        out = target_preexecution.loc[:, list(bridge.IDENTITY)].copy()
        prob_map = dict(zip(("a", "b", "c"), probabilities))
        for col in ("p_uncalibrated", "p_platt", "p_isotonic"):
            out[col] = out["test_id"].map(prob_map).astype(float)
        return out, {"source": "synthetic_fixed_probability_mock"}
    monkeypatch.setattr(bridge, "predict_preselection_rq1", fake_predict)


def _single_run(history, target, oracle, *, include=True, rho=.95):
    return bridge.evaluate_one_build_with_rq1(
        development=pd.DataFrame(), target_preexecution=target,
        history=history, target_oracle=oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=rho,
        rq1_config={}, model_config={}, calibration_config={},
        core_features_config={}, include_h2_budget=include,
    )


def test_h2_optin_uses_same_estimated_cost_cap_and_30_seeded_random(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    result = _single_run(h, t, o)
    assert result.h2_status == "computed"
    assert result.selection_mode == "normal_risk_density"
    pair = result.h2_budget_pairing
    assert pair is not None
    assert pair.random_pair.condition_key == (
        "model=logistic_regression", "probability=uncalibrated", "rho=0.95",
        "cost_policy=historical_median", "observation=retrospective_full_information",
    )
    assert pair.random_pair.build_index == 3
    assert pair.random_pair.reference_selected_test_ids == tuple(sorted(result.selected_test_ids))
    assert pair.random_pair.reference_estimated_cost_ms == pytest.approx(
        result.report["estimated_selected_ms"]
    )
    assert len(pair.random_pair.random_replicates) == 30
    assert all(r.budget_capped.estimated_cost_ms <= pair.random_pair.reference_estimated_cost_ms
               for r in pair.random_pair.random_replicates)
    assert pair.failure_history_budget.estimated_cost_ms <= pair.random_pair.reference_estimated_cost_ms
    assert pair.history_duration_budget.estimated_cost_ms <= pair.random_pair.reference_estimated_cost_ms
    assert pair.h2_primary_rho is True


def test_h2_no_current_oracle_leakage(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    first = _single_run(h, t, o)
    changed = o.copy(deep=True)
    changed["failed"] = [0, 1, 1]
    changed["execution_time_ms"] = [1000, 1001, 1002]
    second = _single_run(h, t, changed)
    assert first.selected_test_ids == second.selected_test_ids
    assert first.h2_budget_pairing == second.h2_budget_pairing
    assert first.report["observed_full_ms"] != second.report["observed_full_ms"]


def test_h2_missing_prior_f1_rate_fails_closed(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    t.loc[t.test_id == "a", "failure_rate_history"] = np.nan
    with pytest.raises(ValueError, match="historical failure rates"):
        _single_run(h, t, o)


def test_h2_inconsistent_unseen_rate_fails_closed(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    t.loc[t.test_id == "c", "failure_rate_history"] = 0.8
    with pytest.raises(ValueError, match="historical failure rates"):
        _single_run(h, t, o)


def test_h2_missing_feature_columns_fail_closed_only_when_opted_in(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    t = t.drop(columns=["failure_rate_history", "prior_execution_count"])
    old = _single_run(h, t, o, include=False)
    assert old.h2_budget_pairing is None
    assert old.h2_status == "not_requested"
    with pytest.raises(ValueError, match="causally materialized F1"):
        _single_run(h, t, o, include=True)


def test_h2_zero_predicted_risk_is_explicitly_not_applicable(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch, probabilities=(0.0, 0.0, 0.0))
    out = _single_run(h, t, o)
    assert out.selection_mode == "fallback_retest_all"
    assert out.h2_status == "not_applicable_zero_total_predicted_risk"
    assert out.h2_budget_pairing is None


def test_h2_absent_prior_positive_cost_is_explicitly_not_applicable(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    t["execution_time_mean_history_ms"] = np.nan
    h["execution_time_ms"] = 0.0
    out = _single_run(h, t, o)
    assert out.selection_mode == "fallback_retest_all"
    assert out.h2_status == "not_applicable_no_positive_prior_cost_history"
    assert out.h2_budget_pairing is None


def test_h2_rejects_noncanonical_rho_if_requested(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    with pytest.raises(ValueError, match="rho condition"):
        _single_run(h, t, o, rho=.92)


def test_h2_cost_alignment_cannot_silently_drift(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    real_estimator = bridge.estimate_build_costs
    def corrupted(*args, **kwargs):
        out = real_estimator(*args, **kwargs)
        out["estimated_cost_ms"] *= 2
        return out
    monkeypatch.setattr(bridge, "estimate_build_costs", corrupted)
    with pytest.raises(RuntimeError, match="cost estimates differ"):
        _single_run(h, t, o)


def test_multibuild_optin_requires_causal_rematerialization():
    with pytest.raises(ValueError, match="causal_from_history"):
        multi.run_synthetic_multibuild_rq1_rq2(
            None, None, None, model_name="logistic_regression",
            probability_variant="uncalibrated", rho=.95,
            rq1_config={}, model_config={}, calibration_config={},
            core_features_config={}, include_h2_budget=True,
        )


def test_multibuild_passes_h2_flag_to_existing_bridge(monkeypatch):
    # Existing in-repo causal fixtures already verify actual feature rematerialization.
    # Here exercise the H2 opt-in path without changing the established runner.
    from types import SimpleNamespace
    project = "synthetic-causal"
    def ts(i):
        return (pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(days=i-1)).isoformat()
    history = pd.DataFrame([
        dict(project_id=project, build_id=f"build-{i}", build_index=i,
             test_id=t, build_started_at=ts(i),
             changed_entity_count=i % 4, failed=int(i % 3 == 0),
             execution_time_ms=12.0 if t == "a" else 20.0)
        for i in range(1, 14) for t in ("a", "b")
    ])
    current = pd.DataFrame([
        dict(project_id=project, build_id="build-14", build_index=14,
             test_id=t, build_started_at=ts(14), changed_entity_count=2)
        for t in ("a", "b")
    ])
    oracle = current.loc[:, list(multi.IDENTITY)].copy()
    oracle["failed"] = [1, 0]
    oracle["execution_time_ms"] = [12.0, 20.0]
    seen = []
    def fake_bridge(*, development, target_preexecution, history, target_oracle, **kwargs):
        seen.append(kwargs.get("include_h2_budget"))
        assert set(pd.to_numeric(history.build_index)) == set(range(1, 14))
        assert "failure_rate_history" in target_preexecution.columns
        return SimpleNamespace(build_index=14, selected_test_ids=("a",), report={})
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", fake_bridge)
    result = multi.run_synthetic_multibuild_rq1_rq2(
        history, current, oracle, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=.95,
        rq1_config={}, model_config={}, calibration_config={},
        core_features_config={}, preexecution_mode="causal_from_history",
        include_h2_budget=True,
    )
    assert seen == [True]
    assert len(result.steps) == 1


def test_h2_new_test_missing_history_rate_is_explicit_zero(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    observed = []
    real_pair = bridge.pair_h2_budget_comparators_for_build
    def checked_pair(**kwargs):
        observed.append(tuple(kwargs["historical_failure_rates"]))
        assert "failed" not in kwargs
        assert "execution_time_ms" not in kwargs
        return real_pair(**kwargs)
    monkeypatch.setattr(bridge, "pair_h2_budget_comparators_for_build", checked_pair)
    _single_run(h, t, o)
    assert observed == [(0.5, 0.0, 0.0)]


def test_h2_future_cost_history_rejected_before_selection(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    h.loc[0, "build_index"] = 3
    with pytest.raises(ValueError, match="target/future"):
        _single_run(h, t, o)


def test_h2_rejects_nonboolean_toggle(monkeypatch):
    h, t, o = _single_frames()
    _mock_predict(monkeypatch)
    with pytest.raises(ValueError, match="include_h2_budget must be boolean"):
        bridge.evaluate_one_build_with_rq1(
            development=pd.DataFrame(), target_preexecution=t, history=h,
            target_oracle=o, model_name="logistic_regression",
            probability_variant="uncalibrated", rho=.95,
            rq1_config={}, model_config={}, calibration_config={},
            core_features_config={}, include_h2_budget="true",
        )
