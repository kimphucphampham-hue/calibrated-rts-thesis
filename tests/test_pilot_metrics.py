import numpy as np

from src.tcp_ci.evaluate_pilot_metrics import (
    calibration_bins,
    evaluate_model,
    expected_calibration_error,
)


def test_perfect_probabilities_have_zero_ece():
    y = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probability = np.array(
        [
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    ece = expected_calibration_error(
        y_true=y,
        probability=probability,
        n_bins=10,
    )

    assert np.isclose(
        ece,
        0.0,
    )


def test_constant_probability_ece_matches_mean_gap():
    y = np.array(
        [
            0,
            0,
            0,
            1,
        ]
    )

    probability = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )

    # Observed prevalence = 0.25
    # Prediction mean = 0.50
    # All rows occupy the same bin.
    expected = 0.25

    actual = expected_calibration_error(
        y_true=y,
        probability=probability,
        n_bins=10,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_calibration_bins_preserve_all_rows():
    y = np.array(
        [
            0,
            1,
            0,
            1,
            0,
        ]
    )

    probability = np.array(
        [
            0.0,
            0.1,
            0.5,
            0.9,
            1.0,
        ]
    )

    bins = calibration_bins(
        y_true=y,
        probability=probability,
        n_bins=10,
    )

    assert (
        bins["count"].sum()
        == len(y)
    )

    assert np.isclose(
        bins["weight"].sum(),
        1.0,
    )


def test_metrics_for_dummy_probability():
    y = np.array(
        [
            0,
            0,
            0,
            1,
        ]
    )

    prevalence = y.mean()

    probability = np.full(
        len(y),
        prevalence,
    )

    metrics, _ = evaluate_model(
        y_true=y,
        probability=probability,
        n_bins=10,
    )

    assert np.isclose(
        metrics["roc_auc"],
        0.5,
    )

    assert np.isclose(
        metrics[
            "average_precision"
        ],
        prevalence,
    )

    assert np.isclose(
        metrics["ap_lift"],
        1.0,
    )

    assert np.isclose(
        metrics[
            "mean_predicted_probability"
        ],
        prevalence,
    )

    assert np.isclose(
        metrics[
            "calibration_bias"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "absolute_calibration_bias"
        ],
        0.0,
    )


def test_invalid_probability_is_rejected():
    y = np.array(
        [
            0,
            1,
        ]
    )

    probability = np.array(
        [
            -0.1,
            1.1,
        ]
    )

    try:
        evaluate_model(
            y_true=y,
            probability=probability,
            n_bins=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid probabilities "
            "should be rejected."
        )


def test_absolute_calibration_bias_removes_direction():
    y = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    # Observed prevalence = 0.50
    # Mean predicted probability = 0.25
    probability = np.array(
        [
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )

    metrics, _ = evaluate_model(
        y_true=y,
        probability=probability,
        n_bins=10,
    )

    assert np.isclose(
        metrics[
            "calibration_bias"
        ],
        -0.25,
    )

    assert np.isclose(
        metrics[
            "absolute_calibration_bias"
        ],
        0.25,
    )