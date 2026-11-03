"""Synthetic condition-isolation tests, not confirmatory RQ2 evidence."""
import pandas as pd
import pytest

from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback
from src.tcp_ci.rq2_cold_start_reporting import (
    GROUP_KEYS, evaluate_post_selection, summarize_selection_reports,
)


def report(build=10, **overrides):
    row = dict(project="synthetic", model="lr", probability="p_platt",
               rho=0.95, build_index=build, selection_mode="normal_risk_density",
               fallback_reason=None, n_tests=2, n_selected=1,
               observed_selected_ms=5.0, observed_full_ms=10.0,
               total_failures=1, captured_failures=1, missed_failures=0,
               rtr_defined=True, failure_metrics_defined=True)
    row.update(overrides)
    return row


def labeled(build=10, **overrides):
    row = report(build, cost_policy="historical_median",
                 baseline="risk_density_sort",
                 observation_regime="retrospective_full_information")
    row.update(overrides)
    return row


@pytest.mark.parametrize("field,alternative,second_build", [
    ("cost_policy", "positive_floor", 11),
    ("cost_policy", "positive_floor", 10),
    ("baseline", "retest_all", 10),
])
def test_distinct_experimental_conditions_are_isolated(field, alternative,
                                                       second_build):
    # Only full-information reports may be compared by policy/baseline.
    second = labeled(second_build)
    second[field] = alternative
    summary = summarize_selection_reports(
        [labeled(10), second], require_condition_metadata=True
    )
    assert len(summary) == 2
    assert set(GROUP_KEYS).issubset(summary.columns)
    assert summary["n_evaluation_builds"].tolist() == [1, 1]


@pytest.mark.parametrize("other", [10, 10.0, "10.0", "1e1", "010"])
def test_duplicate_build_normalization(other):
    with pytest.raises(ValueError, match="Duplicate evaluation build"):
        summarize_selection_reports([labeled("10"), labeled(other)])


@pytest.mark.parametrize("bad", [True, 0, "10.5", "not-a-build", float("inf")])
def test_invalid_build_index_rejected(bad):
    with pytest.raises(ValueError, match="Invalid report build_index"):
        summarize_selection_reports([labeled(bad)])


def test_unlabeled_legacy_reports_remain_supported():
    summary = summarize_selection_reports([report(10), report(11)])
    assert len(summary) == 1
    assert "cost_policy" not in summary.columns
    with pytest.raises(ValueError, match="metadata required"):
        summarize_selection_reports([report(10)], require_condition_metadata=True)


@pytest.mark.parametrize("rows", [
    [labeled(10), report(11)],
    [report(10, cost_policy="historical_median")],
    [labeled(10, baseline="   ")],
])
def test_missing_or_invalid_metadata_rejected(rows):
    with pytest.raises(ValueError, match="experimental condition metadata"):
        summarize_selection_reports(rows)


def test_same_build_and_same_condition_still_rejected():
    one = labeled(10)
    with pytest.raises(ValueError, match="Duplicate evaluation build"):
        summarize_selection_reports([one, dict(one)])


def test_evaluator_optional_metadata_does_not_change_metrics():
    target = pd.DataFrame(dict(
        project_id=["A", "A"], build_index=[10, 10],
        build_id=["b10", "b10"], test_id=["t1", "t2"],
        execution_time_mean_history_ms=[5.0, 100.0],
        failed=[1, 0], execution_time_ms=[5.0, 15.0],
    ))
    history = pd.DataFrame(dict(
        project_id=["A"], build_index=[9], execution_time_ms=[20.0]
    ))
    selection = select_with_cold_start_fallback(
        target=target, history=history, probabilities=[0.9, 0.1], rho=0.8
    )
    old = evaluate_post_selection(target, selection, rho=0.8)
    new = evaluate_post_selection(
        target, selection, rho=0.8, cost_policy=" historical_median ",
        baseline="risk_density_sort",
        observation_regime="retrospective_full_information",
    )
    assert "cost_policy" not in old
    assert new["cost_policy"] == "historical_median"
    for metric in ("n_selected", "rtr", "failure_recall"):
        assert old[metric] == new[metric]
    with pytest.raises(ValueError, match="Incomplete experimental condition"):
        evaluate_post_selection(
            target, selection, rho=0.8, cost_policy="historical_median"
        )

@pytest.mark.parametrize("regime", [
    "partial_observation",
    "synthetic_partial_observation",
    "unknown_regime",
])
def test_evaluator_rejects_unsupported_regime_even_with_complete_truth(regime):
    target = pd.DataFrame({
        "project_id": ["synthetic", "synthetic"],
        "build_index": [10, 10],
        "build_id": ["b10", "b10"],
        "test_id": ["a", "b"],
        "execution_time_mean_history_ms": [5.0, 100.0],
        "failed": [1, 0],
        "execution_time_ms": [5.0, 15.0],
    })
    history = pd.DataFrame({
        "project_id": ["synthetic"],
        "build_index": [9],
        "execution_time_ms": [20.0],
    })
    selection = select_with_cold_start_fallback(
        target=target, history=history, probabilities=[0.9, 0.1], rho=0.8,
    )
    with pytest.raises(ValueError, match="Unsupported observation regime"):
        evaluate_post_selection(
            target, selection, rho=0.8,
            cost_policy="historical_median",
            baseline="risk_density_sort",
            observation_regime=regime,
        )


@pytest.mark.parametrize("regime", [
    "partial_observation", "synthetic_partial_observation",
])
@pytest.mark.parametrize("strict", [False, True])
def test_aggregator_rejects_unsupported_regime_even_with_direct_reports(
    regime, strict,
):
    with pytest.raises(ValueError, match="Unsupported observation regime"):
        summarize_selection_reports(
            [labeled(10, observation_regime=regime)],
            require_condition_metadata=strict,
        )


def test_aggregator_accepts_and_normalizes_full_information_regime():
    summary = summarize_selection_reports(
        [labeled(10, observation_regime=" retrospective_full_information ")],
        require_condition_metadata=True,
    )
    assert summary.iloc[0]["observation_regime"] == (
        "retrospective_full_information"
    )
