from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def validate_probability(
    probability: np.ndarray,
) -> None:
    probability = np.asarray(
        probability,
        dtype=float,
    )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Non-finite probability found."
        )

    if not (
        (
            probability >= 0.0
        )
        &
        (
            probability <= 1.0
        )
    ).all():
        raise ValueError(
            "Probability outside [0, 1]."
        )


def calibration_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    validate_probability(
        probability
    )

    if n_bins <= 1:
        raise ValueError(
            "n_bins must be greater than 1."
        )

    # Equal-width bins:
    # [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_id = np.floor(
        probability * n_bins
    ).astype(int)

    bin_id = np.clip(
        bin_id,
        0,
        n_bins - 1,
    )

    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "probability": probability,
            "bin_id": bin_id,
        }
    )

    rows = []

    total_rows = len(frame)

    for current_bin in range(
        n_bins
    ):
        group = frame[
            frame["bin_id"]
            == current_bin
        ]

        if group.empty:
            continue

        count = len(group)

        mean_probability = float(
            group[
                "probability"
            ].mean()
        )

        observed_rate = float(
            group[
                "y_true"
            ].mean()
        )

        absolute_gap = abs(
            mean_probability
            - observed_rate
        )

        rows.append(
            {
                "bin_id":
                    current_bin,

                "bin_lower":
                    current_bin
                    / n_bins,

                "bin_upper":
                    (
                        current_bin
                        + 1
                    )
                    / n_bins,

                "count":
                    count,

                "weight":
                    count
                    / total_rows,

                "mean_predicted_probability":
                    mean_probability,

                "observed_failure_rate":
                    observed_rate,

                "absolute_gap":
                    absolute_gap,

                "weighted_gap":
                    (
                        count
                        / total_rows
                        * absolute_gap
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise ValueError(
            "No calibration bins produced."
        )

    if int(
        result["count"].sum()
    ) != total_rows:
        raise AssertionError(
            "Calibration-bin row count mismatch."
        )

    if not np.isclose(
        result["weight"].sum(),
        1.0,
    ):
        raise AssertionError(
            "Calibration-bin weights "
            "do not sum to 1."
        )

    return result


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> float:
    bins = calibration_bins(
        y_true=y_true,
        probability=probability,
        n_bins=n_bins,
    )

    return float(
        bins[
            "weighted_gap"
        ].sum()
    )


def evaluate_model(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    validate_probability(
        probability
    )

    classes = set(
        np.unique(
            y_true
        ).tolist()
    )

    if classes != {0, 1}:
        raise ValueError(
            "Evaluation requires both "
            "target classes."
        )

    prevalence = float(
        y_true.mean()
    )

    average_precision = float(
        average_precision_score(
            y_true,
            probability,
        )
    )

    if prevalence <= 0:
        raise ValueError(
            "Failure prevalence must "
            "be positive."
        )

    bins = calibration_bins(
        y_true=y_true,
        probability=probability,
        n_bins=n_bins,
    )

    ece = float(
        bins[
            "weighted_gap"
        ].sum()
    )

    mean_probability = float(
        probability.mean()
    )

    probability_matrix = (
        np.column_stack(
            [
                1.0 - probability,
                probability,
            ]
        )
    )

    metrics = {
        "rows":
            len(y_true),

        "failure_rows":
            int(
                y_true.sum()
            ),

        "failure_prevalence":
            prevalence,

        "mean_predicted_probability":
            mean_probability,

        "calibration_bias":
            (
                mean_probability
                - prevalence
            ),

        "absolute_calibration_bias":
            abs(
                mean_probability
                - prevalence
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            ),

        "average_precision":
            average_precision,

        "ap_lift":
            (
                average_precision
                / prevalence
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probability,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probability_matrix,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "ece":
            ece,
    }

    return (
        metrics,
        bins,
    )
