from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.tcp_ci.probability_metrics import (
    calibration_bins,
    validate_probability,
)

from src.tcp_ci.rq1_pipeline import (
    PROBABILITY_COLUMNS,
    RQ1_CALIBRATION_VARIANTS,
)


AVAILABLE_STATUS = "AVAILABLE"


RQ1_METRIC_COLUMNS = [
    "project_id",
    "model",
    "calibration_variant",
    "condition_status",
    "calibration_support_sufficient",
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


RQ1_BIN_COLUMNS = [
    "project_id",
    "model",
    "calibration_variant",
    "condition_status",
    "bin_id",
    "bin_lower",
    "bin_upper",
    "count",
    "weight",
    "mean_predicted_probability",
    "observed_failure_rate",
    "absolute_gap",
    "weighted_gap",
]


def validate_rq1_target(
    y_true,
) -> np.ndarray:
    y = np.asarray(
        y_true,
        dtype=float,
    )

    if y.ndim != 1:
        raise ValueError(
            "RQ1 evaluation target must "
            "be one-dimensional."
        )

    if y.size == 0:
        raise ValueError(
            "RQ1 evaluation target must "
            "not be empty."
        )

    if not np.isfinite(
        y
    ).all():
        raise ValueError(
            "RQ1 evaluation target must "
            "contain only finite values."
        )

    values = set(
        np.unique(
            y
        ).tolist()
    )

    if not values.issubset(
        {
            0.0,
            1.0,
        }
    ):
        raise ValueError(
            "RQ1 evaluation target must "
            "be binary 0/1."
        )

    return y.astype(
        int
    )


def evaluate_rq1_probability(
    y_true,
    probability,
    n_bins: int,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    y = validate_rq1_target(
        y_true
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if probability.ndim != 1:
        raise ValueError(
            "RQ1 probability must be "
            "one-dimensional."
        )

    if len(
        probability
    ) != len(
        y
    ):
        raise ValueError(
            "RQ1 target and probability "
            "length mismatch."
        )

    validate_probability(
        probability
    )

    prevalence = float(
        y.mean()
    )

    mean_probability = float(
        probability.mean()
    )

    bins = calibration_bins(
        y_true=y,
        probability=probability,
        n_bins=n_bins,
    )

    ece = float(
        bins[
            "weighted_gap"
        ].sum()
    )

    probability_matrix = (
        np.column_stack(
            [
                1.0 - probability,
                probability,
            ]
        )
    )

    classes = set(
        np.unique(
            y
        ).tolist()
    )

    if classes == {
        0,
        1,
    }:
        roc_auc = float(
            roc_auc_score(
                y,
                probability,
            )
        )

        average_precision = float(
            average_precision_score(
                y,
                probability,
            )
        )

    else:
        roc_auc = float(
            "nan"
        )

        average_precision = float(
            "nan"
        )

    metrics = {
        "rows":
            int(
                len(
                    y
                )
            ),

        "failure_rows":
            int(
                y.sum()
            ),

        "failure_prevalence":
            prevalence,

        "roc_auc":
            roc_auc,

        "average_precision":
            average_precision,

        "brier_score":
            float(
                brier_score_loss(
                    y,
                    probability,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    probability_matrix,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "ece":
            ece,

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
    }

    return (
        metrics,
        bins,
    )


def skipped_rq1_metrics(
    y_true,
) -> dict:
    y = validate_rq1_target(
        y_true
    )

    prevalence = float(
        y.mean()
    )

    return {
        "rows":
            int(
                len(
                    y
                )
            ),

        "failure_rows":
            int(
                y.sum()
            ),

        "failure_prevalence":
            prevalence,

        "roc_auc":
            float(
                "nan"
            ),

        "average_precision":
            float(
                "nan"
            ),

        "brier_score":
            float(
                "nan"
            ),

        "log_loss":
            float(
                "nan"
            ),

        "ece":
            float(
                "nan"
            ),

        "mean_predicted_probability":
            float(
                "nan"
            ),

        "calibration_bias":
            float(
                "nan"
            ),

        "absolute_calibration_bias":
            float(
                "nan"
            ),
    }


def evaluate_rq1_predictions(
    predictions: pd.DataFrame,
    diagnostics: dict,
    n_bins: int = 10,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    required_columns = {
        "project_id",
        "failed",
        *PROBABILITY_COLUMNS.values(),
    }

    missing = (
        required_columns
        - set(
            predictions.columns
        )
    )

    if missing:
        raise ValueError(
            "RQ1 predictions are missing "
            f"required columns: "
            f"{sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "RQ1 predictions must not "
            "be empty."
        )

    project_values = (
        predictions[
            "project_id"
        ].drop_duplicates()
    )

    if len(
        project_values
    ) != 1:
        raise ValueError(
            "RQ1 predictions must contain "
            "exactly one project."
        )

    project_id = (
        project_values.iloc[0]
    )

    model_name = str(
        diagnostics[
            "model"
        ]
    )

    support_sufficient = bool(
        diagnostics[
            "calibration_support_sufficient"
        ]
    )

    y_true = predictions[
        "failed"
    ].to_numpy()

    metric_rows = []
    bin_frames = []

    availability = {
        "uncalibrated":
            bool(
                diagnostics[
                    "uncalibrated_available"
                ]
            ),

        "platt":
            bool(
                diagnostics[
                    "platt_available"
                ]
            ),

        "isotonic":
            bool(
                diagnostics[
                    "isotonic_available"
                ]
            ),
    }

    for variant in (
        RQ1_CALIBRATION_VARIANTS
    ):
        probability_column = (
            PROBABILITY_COLUMNS[
                variant
            ]
        )

        available = (
            availability[
                variant
            ]
        )

        probability = predictions[
            probability_column
        ]

        if available:
            if probability.isna().any():
                raise ValueError(
                    f"Available RQ1 variant "
                    f"{variant} contains "
                    "missing probabilities."
                )

            (
                metrics,
                bins,
            ) = evaluate_rq1_probability(
                y_true=y_true,
                probability=
                    probability.to_numpy(),
                n_bins=n_bins,
            )

            condition_status = (
                AVAILABLE_STATUS
            )

            bins = bins.copy()

            bins.insert(
                0,
                "condition_status",
                condition_status,
            )

            bins.insert(
                0,
                "calibration_variant",
                variant,
            )

            bins.insert(
                0,
                "model",
                model_name,
            )

            bins.insert(
                0,
                "project_id",
                project_id,
            )

            bin_frames.append(
                bins[
                    RQ1_BIN_COLUMNS
                ]
            )

        else:
            if not probability.isna().all():
                raise ValueError(
                    f"Skipped RQ1 variant "
                    f"{variant} must contain "
                    "only missing "
                    "probabilities."
                )

            metrics = (
                skipped_rq1_metrics(
                    y_true
                )
            )

            condition_status = str(
                diagnostics[
                    "status"
                ]
            )

        metric_rows.append(
            {
                "project_id":
                    project_id,

                "model":
                    model_name,

                "calibration_variant":
                    variant,

                "condition_status":
                    condition_status,

                "calibration_support_sufficient":
                    support_sufficient,

                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )[
        RQ1_METRIC_COLUMNS
    ]

    if bin_frames:
        bins_df = pd.concat(
            bin_frames,
            ignore_index=True,
        )[
            RQ1_BIN_COLUMNS
        ]

    else:
        bins_df = pd.DataFrame(
            columns=
                RQ1_BIN_COLUMNS
        )

    if len(
        metrics_df
    ) != len(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise AssertionError(
            "RQ1 evaluator must produce "
            "exactly one metric row per "
            "calibration variant."
        )

    return (
        metrics_df,
        bins_df,
    )
