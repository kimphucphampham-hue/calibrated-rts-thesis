import numpy as np
import pandas as pd
import pytest

from src.tcp_ci.probability_metrics import (
    evaluate_model,
)

from src.tcp_ci.rq1_evaluator import (
    AVAILABLE_STATUS,
    RQ1_METRIC_COLUMNS,
    evaluate_rq1_predictions,
    evaluate_rq1_probability,
)


def test_two_class_rq1_metrics_match_existing_probability_metrics():
    y = np.array(
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

    rq1_metrics, rq1_bins = (
        evaluate_rq1_probability(
            y_true=y,
            probability=probability,
            n_bins=10,
        )
    )

    pilot_metrics, pilot_bins = (
        evaluate_model(
            y_true=y,
            probability=probability,
            n_bins=10,
        )
    )

    comparable_metrics = [
        "rows",
        "failure_rows",
        "failure_prevalence",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ece",
        "mean_predicted_probability",
        "calibration_bias",
        "absolute_calibration_bias",
    ]

    for name in comparable_metrics:
        assert np.isclose(
            rq1_metrics[
                name
            ],
            pilot_metrics[
                name
            ],
        )

    pd.testing.assert_frame_equal(
        rq1_bins,
        pilot_bins,
    )


def test_single_negative_class_reports_discrimination_as_na():
    y = np.array(
        [
            0,
            0,
            0,
        ],
        dtype=int,
    )

    probability = np.array(
        [
            0.1,
            0.2,
            0.3,
        ],
        dtype=float,
    )

    metrics, bins = (
        evaluate_rq1_probability(
            y_true=y,
            probability=probability,
            n_bins=10,
        )
    )

    assert np.isnan(
        metrics[
            "roc_auc"
        ]
    )

    assert np.isnan(
        metrics[
            "average_precision"
        ]
    )

    for name in [
        "brier_score",
        "log_loss",
        "ece",
        "mean_predicted_probability",
        "calibration_bias",
        "absolute_calibration_bias",
    ]:
        assert np.isfinite(
            metrics[
                name
            ]
        )

    assert (
        int(
            bins[
                "count"
            ].sum()
        )
        == len(
            y
        )
    )


def test_single_positive_class_reports_discrimination_as_na():
    y = np.array(
        [
            1,
            1,
            1,
        ],
        dtype=int,
    )

    probability = np.array(
        [
            0.6,
            0.8,
            0.9,
        ],
        dtype=float,
    )

    metrics, _ = (
        evaluate_rq1_probability(
            y_true=y,
            probability=probability,
            n_bins=10,
        )
    )

    assert np.isnan(
        metrics[
            "roc_auc"
        ]
    )

    assert np.isnan(
        metrics[
            "average_precision"
        ]
    )

    assert np.isfinite(
        metrics[
            "brier_score"
        ]
    )

    assert np.isfinite(
        metrics[
            "log_loss"
        ]
    )

    assert np.isfinite(
        metrics[
            "ece"
        ]
    )


def make_sufficient_predictions():
    return pd.DataFrame(
        {
            "project_id": [
                "p",
                "p",
                "p",
                "p",
            ],

            "failed": [
                0,
                0,
                1,
                1,
            ],

            "p_uncalibrated": [
                0.1,
                0.2,
                0.7,
                0.9,
            ],

            "p_platt": [
                0.05,
                0.15,
                0.75,
                0.95,
            ],

            "p_isotonic": [
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        }
    )


def sufficient_diagnostics():
    return {
        "model":
            "logistic_regression",

        "status":
            "SUFFICIENT",

        "calibration_support_sufficient":
            True,

        "uncalibrated_available":
            True,

        "platt_available":
            True,

        "isotonic_available":
            True,
    }


def test_sufficient_predictions_produce_three_conditions():
    (
        metrics,
        bins,
    ) = evaluate_rq1_predictions(
        predictions=
            make_sufficient_predictions(),
        diagnostics=
            sufficient_diagnostics(),
        n_bins=10,
    )

    assert list(
        metrics.columns
    ) == RQ1_METRIC_COLUMNS

    assert (
        len(
            metrics
        )
        == 3
    )

    assert metrics[
        "calibration_variant"
    ].tolist() == [
        "uncalibrated",
        "platt",
        "isotonic",
    ]

    assert set(
        metrics[
            "condition_status"
        ]
    ) == {
        AVAILABLE_STATUS,
    }

    assert set(
        bins[
            "calibration_variant"
        ].unique()
    ) == {
        "uncalibrated",
        "platt",
        "isotonic",
    }


def test_insufficient_predictions_keep_three_metric_conditions():
    predictions = pd.DataFrame(
        {
            "project_id": [
                "p",
                "p",
                "p",
            ],

            "failed": [
                0,
                1,
                0,
            ],

            "p_uncalibrated": [
                0.1,
                0.8,
                0.2,
            ],

            "p_platt": [
                np.nan,
                np.nan,
                np.nan,
            ],

            "p_isotonic": [
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    diagnostics = {
        "model":
            "logistic_regression",

        "status":
            "INSUFFICIENT_CALIBRATION_SUPPORT",

        "calibration_support_sufficient":
            False,

        "uncalibrated_available":
            True,

        "platt_available":
            False,

        "isotonic_available":
            False,
    }

    (
        metrics,
        bins,
    ) = evaluate_rq1_predictions(
        predictions=predictions,
        diagnostics=diagnostics,
        n_bins=10,
    )

    assert (
        len(
            metrics
        )
        == 3
    )

    raw = metrics[
        metrics[
            "calibration_variant"
        ]
        == "uncalibrated"
    ].iloc[0]

    platt = metrics[
        metrics[
            "calibration_variant"
        ]
        == "platt"
    ].iloc[0]

    isotonic = metrics[
        metrics[
            "calibration_variant"
        ]
        == "isotonic"
    ].iloc[0]

    assert (
        raw[
            "condition_status"
        ]
        == AVAILABLE_STATUS
    )

    assert np.isfinite(
        raw[
            "brier_score"
        ]
    )

    for row in [
        platt,
        isotonic,
    ]:
        assert (
            row[
                "condition_status"
            ]
            ==
            "INSUFFICIENT_CALIBRATION_SUPPORT"
        )

        for metric in [
            "roc_auc",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece",
            "mean_predicted_probability",
            "calibration_bias",
            "absolute_calibration_bias",
        ]:
            assert pd.isna(
                row[
                    metric
                ]
            )

    assert set(
        bins[
            "calibration_variant"
        ].unique()
    ) == {
        "uncalibrated",
    }


def test_available_variant_rejects_partial_missing_probability():
    predictions = (
        make_sufficient_predictions()
    )

    predictions.loc[
        predictions.index[0],
        "p_platt",
    ] = np.nan

    try:
        evaluate_rq1_predictions(
            predictions=predictions,
            diagnostics=
                sufficient_diagnostics(),
            n_bins=10,
        )

    except ValueError as error:
        assert (
            "missing probabilities"
            in str(
                error
            )
        )

    else:
        raise AssertionError(
            "Partially missing available "
            "probability must be rejected."
        )


def test_skipped_variant_rejects_non_missing_probability():
    predictions = pd.DataFrame(
        {
            "project_id": [
                "p",
                "p",
            ],

            "failed": [
                0,
                1,
            ],

            "p_uncalibrated": [
                0.1,
                0.8,
            ],

            "p_platt": [
                np.nan,
                0.5,
            ],

            "p_isotonic": [
                np.nan,
                np.nan,
            ],
        }
    )

    diagnostics = {
        "model":
            "logistic_regression",

        "status":
            "INSUFFICIENT_CALIBRATION_SUPPORT",

        "calibration_support_sufficient":
            False,

        "uncalibrated_available":
            True,

        "platt_available":
            False,

        "isotonic_available":
            False,
    }

    try:
        evaluate_rq1_predictions(
            predictions=predictions,
            diagnostics=diagnostics,
            n_bins=10,
        )

    except ValueError as error:
        assert (
            "only missing probabilities"
            in str(
                error
            )
        )

    else:
        raise AssertionError(
            "Skipped calibration variant "
            "with a probability must be "
            "rejected."
        )


@pytest.mark.parametrize(
    "target",
    [
        np.array(
            [
                0.0,
                0.5,
                1.0,
            ]
        ),
        np.array(
            [
                -0.2,
                0.0,
                1.0,
            ]
        ),
        np.array(
            [
                0.0,
                1.0,
                1.8,
            ]
        ),
    ],
)
def test_rq1_evaluator_rejects_non_binary_numeric_target(
    target,
):
    probability = np.full(
        len(
            target
        ),
        0.5,
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="binary 0/1",
    ):
        evaluate_rq1_probability(
            y_true=target,
            probability=probability,
            n_bins=10,
        )
