"""Tests for candidate M2-06 post-selection reporting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.tcp_ci.rq2_cold_start_orchestrator import (
    select_with_cold_start_fallback,
)

from src.tcp_ci.rq2_cold_start_reporting import (
    evaluate_post_selection,
    summarize_selection_reports,
)


def make_target(
    costs,
    build_index=10,
    failures=(1, 0),
    durations=(5.0, 15.0),
):
    return pd.DataFrame({
        "project_id": ["project-A", "project-A"],
        "build_index": [build_index, build_index],
        "build_id": [
            f"build-{build_index}",
            f"build-{build_index}",
        ],
        "test_id": ["test-a", "test-b"],
        "execution_time_mean_history_ms": costs,
        "failed": failures,
        "execution_time_ms": durations,
    }, index=[20, 21])


def make_history(positive=False):
    return pd.DataFrame({
        "project_id": ["project-A"],
        "build_index": [9],
        "execution_time_ms": [
            20.0 if positive else 0.0
        ],
    })


def select_and_report(
    target,
    *,
    rho=0.8,
    history=None,
):
    if history is None:
        history = make_history()

    selection = select_with_cold_start_fallback(
        target=target,
        history=history,
        probabilities=[0.9, 0.1],
        rho=rho,
    )

    return evaluate_post_selection(
        target=target,
        selection=selection,
        rho=rho,
        model="synthetic_model",
        probability="synthetic_probability",
    )


def test_normal_selection_metrics():
    result = select_and_report(
        make_target([5.0, 100.0])
    )

    assert result["selection_mode"] == (
        "normal_risk_density"
    )

    assert result["is_fallback"] is False
    assert result["n_tests"] == 2
    assert result["n_selected"] == 1

    assert result["tsr"] == pytest.approx(0.5)
    assert result["rtr"] == pytest.approx(0.25)

    assert result["total_failures"] == 1
    assert result["captured_failures"] == 1
    assert result["missed_failures"] == 0

    assert result["failure_recall"] == pytest.approx(
        1.0
    )

    assert result["missed_failure_rate"] == pytest.approx(
        0.0
    )

    assert result["estimated_selected_ms"] == pytest.approx(
        5.0
    )

    assert result["estimated_cost_defined"] is True


def test_fallback_retest_all_metrics():
    result = select_and_report(
        make_target([0.0, np.nan])
    )

    assert result["selection_mode"] == (
        "fallback_retest_all"
    )

    assert result["fallback_reason"] == (
        "no_positive_prior_cost_history"
    )

    assert result["n_tests"] == 2
    assert result["n_selected"] == 2

    assert result["tsr"] == pytest.approx(1.0)
    assert result["rtr"] == pytest.approx(1.0)

    assert result["failure_recall"] == pytest.approx(
        1.0
    )

    assert result["missed_failure_rate"] == pytest.approx(
        0.0
    )

    assert result["estimated_selected_ms"] is None
    assert result["estimated_cost_defined"] is False

    assert result["selected_test_ids"] == (
        "test-a|test-b"
    )


def test_no_failure_build_has_undefined_failure_metrics():
    result = select_and_report(
        make_target(
            [0.0, np.nan],
            failures=(0, 0),
        )
    )

    assert result["total_failures"] == 0
    assert np.isnan(result["failure_recall"])
    assert np.isnan(result["missed_failure_rate"])

    assert result["failure_metrics_defined"] is False
    assert result["tsr"] == pytest.approx(1.0)


def test_zero_runtime_build_has_undefined_rtr():
    result = select_and_report(
        make_target(
            [0.0, np.nan],
            durations=(0.0, 0.0),
        )
    )

    assert result["observed_full_ms"] == 0
    assert result["observed_selected_ms"] == 0

    assert np.isnan(result["rtr"])
    assert result["rtr_defined"] is False


def test_incomplete_fallback_selection_is_rejected():
    target = make_target([0.0, np.nan])

    selection = select_with_cold_start_fallback(
        target,
        make_history(),
        probabilities=[0.9, 0.1],
        rho=0.8,
    )

    invalid = replace(
        selection,
        selected_test_ids=("test-a",),
    )

    with pytest.raises(
        ValueError,
        match="Invalid fallback selection diagnostic",
    ):
        evaluate_post_selection(
            target, invalid, rho=0.8
        )


def test_unknown_selected_ids_are_rejected():
    target = make_target([5.0, 100.0])

    selection = select_with_cold_start_fallback(
        target,
        make_history(),
        probabilities=[0.9, 0.1],
        rho=0.8,
    )

    invalid = replace(
        selection,
        selected_test_ids=("unknown-test",),
    )

    with pytest.raises(
        ValueError,
        match="outside evaluation target",
    ):
        evaluate_post_selection(
            target, invalid, rho=0.8
        )


def test_nonbinary_failure_labels_are_rejected():
    target = make_target(
        [0.0, np.nan],
        failures=(1, 2),
    )

    with pytest.raises(
        ValueError,
        match="failed must contain only 0 or 1",
    ):
        select_and_report(target)


def test_negative_realized_duration_is_rejected():
    target = make_target(
        [0.0, np.nan],
        durations=(-5.0, 15.0),
    )

    with pytest.raises(
        ValueError,
        match="Negative execution_time_ms",
    ):
        select_and_report(target)


def test_build_identity_mismatch_is_rejected():
    target = make_target([5.0, 100.0])

    selection = select_with_cold_start_fallback(
        target,
        make_history(),
        probabilities=[0.9, 0.1],
        rho=0.8,
    )

    invalid = replace(
        selection,
        build_index="999",
    )

    with pytest.raises(
        ValueError,
        match="build identity mismatch",
    ):
        evaluate_post_selection(
            target, invalid, rho=0.8
        )


def test_summary_keeps_fallback_and_normal_builds():
    normal = select_and_report(
        make_target(
            [5.0, 100.0],
            build_index=10,
        )
    )

    fallback = select_and_report(
        make_target(
            [0.0, np.nan],
            build_index=11,
            failures=(0, 1),
            durations=(0.0, 30.0),
        )
    )

    summary = summarize_selection_reports(
        [normal, fallback]
    )

    assert len(summary) == 1

    result = summary.iloc[0]

    assert result["n_evaluation_builds"] == 2
    assert result["n_normal_builds"] == 1
    assert result["n_fallback_builds"] == 1

    assert result["n_tests"] == 4
    assert result["n_selected"] == 3

    assert result["tsr_micro"] == pytest.approx(
        0.75
    )

    assert result["rtr_micro"] == pytest.approx(
        35.0 / 50.0
    )

    assert result["total_failures"] == 2
    assert result["captured_failures"] == 2

    assert result["failure_recall_micro"] == pytest.approx(
        1.0
    )

    assert result["missed_failure_rate_micro"] == pytest.approx(
        0.0
    )


def test_summary_keeps_undefined_metrics():
    first = select_and_report(
        make_target(
            [0.0, np.nan],
            build_index=10,
            failures=(0, 0),
            durations=(0.0, 0.0),
        )
    )

    second = select_and_report(
        make_target(
            [0.0, np.nan],
            build_index=11,
            failures=(0, 0),
            durations=(0.0, 0.0),
        )
    )

    summary = summarize_selection_reports(
        [first, second]
    )

    result = summary.iloc[0]

    assert result["n_evaluation_builds"] == 2
    assert result["n_fallback_builds"] == 2

    assert result["n_undefined_rtr_builds"] == 2
    assert result["n_zero_failure_builds"] == 2

    assert np.isnan(result["rtr_micro"])
    assert np.isnan(result["failure_recall_micro"])
    assert np.isnan(result["missed_failure_rate_micro"])


def test_duplicate_evaluation_build_is_rejected():
    report = select_and_report(
        make_target([0.0, np.nan])
    )

    with pytest.raises(
        ValueError,
        match="Duplicate evaluation build",
    ):
        summarize_selection_reports(
            [report, report]
        )


def test_different_rho_conditions_are_not_pooled():
    report = select_and_report(
        make_target([0.0, np.nan]),
        rho=0.8,
    )

    alternative = dict(report)
    alternative["rho"] = 0.95

    summary = summarize_selection_reports(
        [report, alternative]
    )

    assert len(summary) == 2
    assert set(summary["rho"]) == {0.8, 0.95}
