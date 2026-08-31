import pandas as pd

from src.tcp_ci.export_project_outputs import (
    build_project_summary,
)


def make_history_row():
    return pd.Series(
        {
            "rows": 100,
            "unique_builds": 20,
            "unique_tests": 10,
            "cold_start_rows": 10,
            "history_rows": 90,
            "history_rate": 0.9,
        }
    )


def make_split_row():
    return pd.Series(
        {
            "total_rows": 100,
            "total_builds": 20,
            "train_rows": 80,
            "test_rows": 20,
            "train_builds": 16,
            "test_builds": 4,
            "train_failure_rate": 0.05,
            "test_failure_rate": 0.10,
            "train_last_build_index": 16,
            "test_first_build_index": 17,
        }
    )


def make_metrics():
    return pd.DataFrame(
        [
            {
                "project_key": "p",
                "model": "dummy_prior",
                "roc_auc": 0.5,
                "average_precision": 0.1,
                "ap_lift": 1.0,
                "brier_score": 0.09,
                "log_loss": 0.3,
                "ece": 0.02,
                "calibration_bias": -0.02,
                "absolute_calibration_bias": 0.02,
            },
            {
                "project_key": "p",
                "model": "logistic_regression",
                "roc_auc": 0.9,
                "average_precision": 0.5,
                "ap_lift": 5.0,
                "brier_score": 0.04,
                "log_loss": 0.2,
                "ece": 0.01,
                "calibration_bias": -0.01,
                "absolute_calibration_bias": 0.01,
            },
        ]
    )


def test_project_summary_contains_core_counts():
    summary = build_project_summary(
        project_key="p",
        history_row=make_history_row(),
        split_row=make_split_row(),
        project_metrics=make_metrics(),
    )

    assert (
        summary["project_key"]
        == "p"
    )

    assert (
        summary["history"]["rows"]
        == 100
    )

    assert (
        summary["history"]["unique_builds"]
        == 20
    )

    assert (
        summary[
            "temporal_split"
        ][
            "train_builds"
        ]
        == 16
    )

    assert (
        summary[
            "temporal_split"
        ][
            "test_builds"
        ]
        == 4
    )


def test_project_summary_contains_logistic_metrics():
    summary = build_project_summary(
        project_key="p",
        history_row=make_history_row(),
        split_row=make_split_row(),
        project_metrics=make_metrics(),
    )

    logistic = summary[
        "logistic_regression"
    ]

    assert (
        logistic["roc_auc"]
        == 0.9
    )

    assert (
        logistic[
            "average_precision"
        ]
        == 0.5
    )

    assert (
        logistic["ece"]
        == 0.01
    )

    assert (
        logistic[
            "absolute_calibration_bias"
        ]
        == 0.01
    )


def test_project_summary_contains_dummy_metrics():
    summary = build_project_summary(
        project_key="p",
        history_row=make_history_row(),
        split_row=make_split_row(),
        project_metrics=make_metrics(),
    )

    dummy = summary[
        "dummy_prior"
    ]

    assert (
        dummy["roc_auc"]
        == 0.5
    )

    assert (
        dummy[
            "average_precision"
        ]
        == 0.1
    )


def test_project_summary_rejects_missing_model():
    metrics = make_metrics()

    metrics = metrics[
        metrics["model"]
        != "logistic_regression"
    ]

    try:
        build_project_summary(
            project_key="p",
            history_row=make_history_row(),
            split_row=make_split_row(),
            project_metrics=metrics,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Missing logistic model "
            "should be rejected."
        )