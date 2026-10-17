import numpy as np

from src.tcp_ci import (
    evaluate_pilot_metrics,
)

from src.tcp_ci.probability_metrics import (
    calibration_bins,
    evaluate_model,
    expected_calibration_error,
)


def test_pilot_evaluator_reexports_shared_metrics():
    assert (
        evaluate_pilot_metrics.calibration_bins
        is calibration_bins
    )

    assert (
        evaluate_pilot_metrics.expected_calibration_error
        is expected_calibration_error
    )

    assert (
        evaluate_pilot_metrics.evaluate_model
        is evaluate_model
    )


def test_shared_metric_output_schema_matches_pilot():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=int,
    )

    probability = np.array(
        [
            0.1,
            0.2,
            0.7,
            0.9,
        ],
        dtype=float,
    )

    metrics, bins = evaluate_model(
        y_true=y_true,
        probability=probability,
        n_bins=10,
    )

    assert set(
        metrics.keys()
    ) == {
        "rows",
        "failure_rows",
        "failure_prevalence",
        "mean_predicted_probability",
        "calibration_bias",
        "absolute_calibration_bias",
        "roc_auc",
        "average_precision",
        "ap_lift",
        "brier_score",
        "log_loss",
        "ece",
    }

    assert int(
        bins["count"].sum()
    ) == len(y_true)


def test_probability_one_is_assigned_to_final_bin():
    y_true = np.array(
        [
            0,
            1,
        ],
        dtype=int,
    )

    probability = np.array(
        [
            0.0,
            1.0,
        ],
        dtype=float,
    )

    bins = calibration_bins(
        y_true=y_true,
        probability=probability,
        n_bins=10,
    )

    assert set(
        bins["bin_id"].tolist()
    ) == {
        0,
        9,
    }

    final_bin = bins[
        bins["bin_id"] == 9
    ].iloc[0]

    assert (
        final_bin["bin_upper"]
        == 1.0
    )
