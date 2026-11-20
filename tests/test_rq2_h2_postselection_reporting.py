"""Synthetic-only post-selection H2 scoring and project aggregation tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci import rq1_rq2_single_build_bridge as bridge
from src.tcp_ci.rq2_h2_postselection_reporting import (
    EVIDENCE_SCOPE, report_h2_prototype_steps,
)


def _frames():
    h = pd.DataFrame([
        dict(project_id="synthetic-e3", build_index=i, test_id=t,
             execution_time_ms=cost)
        for i in (1, 2) for t, cost in (("a", 10.0), ("b", 5.0))
    ])
    t = pd.DataFrame([
        dict(project_id="synthetic-e3", build_id="build-3", build_index=3,
             test_id=k, execution_time_mean_history_ms=cost,
             prior_execution_count=n, failure_rate_history=rate)
        for k, cost, n, rate in (("a", 10.0, 2, .5), ("b", 5.0, 2, 0.0),
                                  ("c", np.nan, 0, np.nan))
    ])
    oracle = t.loc[:, list(bridge.IDENTITY)].copy()
    oracle["failed"] = [1, 0, 1]
    oracle["execution_time_ms"] = [13., 7., 17.]
    return h, t, oracle


def _step(monkeypatch, *, probabilities=(.90, .06, .04), include=True,
          history=None, target=None, oracle=None):
    h, t, o = _frames()
    h = h if history is None else history
    t = t if target is None else target
    o = o if oracle is None else oracle

    def fake_predict(*, development, target_preexecution, **kwargs):
        assert "failed" not in target_preexecution
        assert "execution_time_ms" not in target_preexecution
        out = target_preexecution.loc[:, list(bridge.IDENTITY)].copy()
        p = dict(zip(("a", "b", "c"), probabilities))
        for name in ("p_uncalibrated", "p_platt", "p_isotonic"):
            out[name] = out["test_id"].map(p).astype(float)
        return out, {"source": "synthetic_probability_fixture"}

    monkeypatch.setattr(bridge, "predict_preselection_rq1", fake_predict)
    result = bridge.evaluate_one_build_with_rq1(
        development=pd.DataFrame(), target_preexecution=t,
        history=h, target_oracle=o, model_name="logistic_regression",
        probability_variant="uncalibrated", rho=.95,
        rq1_config={}, model_config={}, calibration_config={},
        core_features_config={}, include_h2_budget=include,
    )
    return result, o


def test_postselection_score_outputs_all_strata(monkeypatch):
    step, oracle = _step(monkeypatch)
    result = report_h2_prototype_steps([step], oracle)
    assert len(result.build_status) == 1
    assert result.build_status.iloc[0].h2_status == "computed"
    assert len(result.build_metrics) == 64  # reference + 2 history + full + 30 budget + 30 count
    assert len(result.project_metrics) == 64
    assert set(result.build_metrics.evidence_scope) == {EVIDENCE_SCOPE}
    primary = result.build_metrics.query("comparison_role in ['reference', 'budget_comparator']")
    assert len(primary) == 33
    assert (primary.estimated_selected_ms <= primary.budget_cap_ms + 1e-7).all()
    assert len(result.build_metrics.query("baseline == 'random_matched_budget'")) == 30
    assert set(result.build_metrics.query("baseline == 'random_matched_budget'").repetition) == set(range(30))
    assert all(result.project_metrics.n_input_builds == 1)
    assert all(result.project_metrics.n_h2_computed_builds == 1)


def test_primary_reference_accounting_and_undefined_denominators(monkeypatch):
    step, oracle = _step(monkeypatch)
    result = report_h2_prototype_steps([step], oracle)
    ref = result.build_metrics.query("baseline == 'risk_density_reference'").iloc[0]
    assert ref.n_tests == 3
    assert ref.n_selected == len(step.selected_test_ids)
    assert ref.total_failures == 2
    assert ref.failure_recall == pytest.approx(ref.captured_failures / 2)
    assert ref.missed_failure_rate == pytest.approx(ref.missed_failures / 2)
    assert ref.budget_utilization == pytest.approx(1.)
    assert ref.observed_full_ms == pytest.approx(37.)
    oracle["failed"] = 0
    oracle["execution_time_ms"] = 0.0
    zero = report_h2_prototype_steps([step], oracle)
    assert not zero.build_metrics.rtr_defined.any()
    assert not zero.build_metrics.failure_metrics_defined.any()
    assert zero.build_metrics.rtr.isna().all()
    assert zero.build_metrics.failure_recall.isna().all()
    assert zero.project_metrics.rtr_micro.isna().all()
    assert zero.project_metrics.missed_failure_rate_micro.isna().all()
    assert zero.project_metrics.n_zero_failure_builds.eq(1).all()


def test_changing_oracle_changes_scoring_not_fixed_choices(monkeypatch):
    step, oracle = _step(monkeypatch)
    first = report_h2_prototype_steps([step], oracle)
    oracle["failed"] = [0, 1, 0]
    oracle["execution_time_ms"] = [100., 500., 1200.]
    second = report_h2_prototype_steps([step], oracle)
    cols = ["baseline", "comparison_role", "repetition", "selected_test_ids", "estimated_selected_ms"]
    pd.testing.assert_frame_equal(first.build_metrics[cols], second.build_metrics[cols])
    assert (first.build_metrics.observed_full_ms != second.build_metrics.observed_full_ms).all()


def test_supplementary_methods_are_labeled_not_budget_comparable(monkeypatch):
    step, oracle = _step(monkeypatch)
    x = report_h2_prototype_steps([step], oracle).build_metrics
    assert set(x.loc[x.baseline.eq("retest_all"), "comparison_role"]) == {"supplementary_full_retest"}
    assert set(x.loc[x.baseline.eq("random_matched_count"), "comparison_role"]) == {"supplementary_matched_count"}
    assert len(x.query("comparison_role == 'supplementary_matched_count'")) == 30


def test_fallback_retains_status_without_fake_budget(monkeypatch):
    step, oracle = _step(monkeypatch, probabilities=(0, 0, 0))
    reports = report_h2_prototype_steps([step], oracle)
    assert step.h2_status == "not_applicable_zero_total_predicted_risk"
    assert reports.build_metrics.empty and reports.project_metrics.empty
    assert reports.build_status.iloc[0].h2_status == step.h2_status


def test_absent_cost_history_fallback_status(monkeypatch):
    h, t, o = _frames()
    h["execution_time_ms"] = 0.0
    t["execution_time_mean_history_ms"] = np.nan
    step, _ = _step(monkeypatch, history=h, target=t, oracle=o)
    reports = report_h2_prototype_steps([step], o)
    assert step.h2_status == "not_applicable_no_positive_prior_cost_history"
    assert reports.build_metrics.empty
    assert len(reports.build_status) == 1


def test_no_h2_request_keeps_only_diagnostic_status(monkeypatch):
    step, oracle = _step(monkeypatch, include=False)
    reports = report_h2_prototype_steps([step], oracle)
    assert reports.build_metrics.empty
    assert reports.build_status.iloc[0].h2_status == "not_requested"


def test_row_permutation_invariant(monkeypatch):
    step, oracle = _step(monkeypatch)
    left = report_h2_prototype_steps([step], oracle)
    right = report_h2_prototype_steps([step], oracle.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(left.build_metrics, right.build_metrics)
    pd.testing.assert_frame_equal(left.project_metrics, right.project_metrics)


def test_duplicate_build_step_refused(monkeypatch):
    step, oracle = _step(monkeypatch)
    with pytest.raises(ValueError, match="Duplicate H2 build"):
        report_h2_prototype_steps([step, step], oracle)


def test_nonexistent_oracle_test_refused(monkeypatch):
    step, oracle = _step(monkeypatch)
    with pytest.raises(ValueError, match="candidate count mismatch"):
        report_h2_prototype_steps([step], oracle.iloc[:2])


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_invalid_oracle_duration_fail_closed(monkeypatch, value):
    step, oracle = _step(monkeypatch)
    oracle.loc[0, "execution_time_ms"] = value
    with pytest.raises(ValueError):
        report_h2_prototype_steps([step], oracle)


@pytest.mark.parametrize("value", [np.nan, 2, -1])
def test_invalid_oracle_failure_fail_closed(monkeypatch, value):
    step, oracle = _step(monkeypatch)
    oracle.loc[0, "failed"] = value
    with pytest.raises(ValueError):
        report_h2_prototype_steps([step], oracle)


def test_reference_cost_drift_refused(monkeypatch):
    step, oracle = _step(monkeypatch)
    altered = replace(step, report={**step.report, "estimated_selected_ms": 999999.})
    with pytest.raises(ValueError, match="reference cost"):
        report_h2_prototype_steps([altered], oracle)


def test_bad_paired_identity_refused(monkeypatch):
    step, oracle = _step(monkeypatch)
    wrong = replace(step.h2_budget_pairing.random_pair, project_id="some-other-project")
    altered = replace(step, h2_budget_pairing=replace(step.h2_budget_pairing, random_pair=wrong))
    with pytest.raises(ValueError, match="H2 comparator"):
        report_h2_prototype_steps([altered], oracle)


def test_refuses_duplicate_oracle_identity(monkeypatch):
    step, oracle = _step(monkeypatch)
    oracle = pd.concat([oracle, oracle.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate H2 oracle"):
        report_h2_prototype_steps([step], oracle)


def test_empty_steps_refused(monkeypatch):
    _, oracle = _step(monkeypatch)
    with pytest.raises(ValueError, match="Nonempty"):
        report_h2_prototype_steps([], oracle)


def test_unknown_h2_status_refused(monkeypatch):
    step, oracle = _step(monkeypatch)
    with pytest.raises(ValueError, match="Unknown H2"):
        report_h2_prototype_steps([replace(step, h2_status="approved")], oracle)
