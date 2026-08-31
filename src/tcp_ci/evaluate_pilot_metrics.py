from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_CONFIG = Path(
    "configs/pilot_metrics_v1.yaml"
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def safe_project_name(
    project_key: str,
) -> str:
    return (
        project_key
        .replace("/", "__")
        .replace("@", "__")
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


def run_project(
    project_key: str,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    results_root = Path(
        config[
            "inputs"
        ][
            "baseline_results_root"
        ]
    )

    prediction_path = (
        results_root
        / safe_project_name(
            project_key
        )
        / "test_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            prediction_path
        )

    target = config[
        "target"
    ][
        "field"
    ]

    models = config[
        "models"
    ]

    probability_fields = [
        definition[
            "probability_field"
        ]
        for definition
        in models.values()
    ]

    required = [
        "project_id",
        "build_id",
        "build_index",
        "test_id",
        target,
        *probability_fields,
    ]

    dataframe = pd.read_csv(
        prediction_path,
        usecols=required,
        dtype={
            "project_id": "string",
            "build_id": "string",
            "test_id": "string",
        },
        low_memory=False,
    )

    project_values = (
        dataframe[
            "project_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if project_values != [
        project_key
    ]:
        raise ValueError(
            f"{project_key}: unexpected "
            f"project_id values: "
            f"{project_values}"
        )

    y_true = (
        pd.to_numeric(
            dataframe[target],
            errors="raise",
        )
        .astype("int8")
        .to_numpy()
    )

    n_bins = int(
        config[
            "ece"
        ][
            "n_bins"
        ]
    )

    metric_rows = []
    bin_frames = []

    for (
        model_name,
        definition,
    ) in models.items():
        probability_field = (
            definition[
                "probability_field"
            ]
        )

        probability = (
            pd.to_numeric(
                dataframe[
                    probability_field
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=float
            )
        )

        (
            metrics,
            bins,
        ) = evaluate_model(
            y_true=y_true,
            probability=probability,
            n_bins=n_bins,
        )

        metric_rows.append(
            {
                "project_key":
                    project_key,

                "model":
                    model_name,

                **metrics,
            }
        )

        bins.insert(
            0,
            "model",
            model_name,
        )

        bins.insert(
            0,
            "project_key",
            project_key,
        )

        bin_frames.append(
            bins
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    bins_df = pd.concat(
        bin_frames,
        ignore_index=True,
    )

    return (
        metrics_df,
        bins_df,
    )


def validate_against_baseline(
    metrics: pd.DataFrame,
    baseline_root: Path,
) -> None:
    baseline_path = (
        baseline_root
        / "baseline_summary.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            baseline_path
        )

    baseline = pd.read_csv(
        baseline_path
    )

    common_metrics = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ]

    left = (
        metrics[
            [
                "project_key",
                "model",
                *common_metrics,
            ]
        ]
        .sort_values(
            [
                "project_key",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    right = (
        baseline[
            [
                "project_key",
                "model",
                *common_metrics,
            ]
        ]
        .sort_values(
            [
                "project_key",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    pilot_config = load_yaml(
        Path(
            config[
                "inputs"
            ][
                "pilot_projects_config"
            ]
        )
    )

    projects = [
        item["project_key"]
        for item
        in pilot_config[
            "projects"
        ]
    ]

    output_root = Path(
        config[
            "outputs"
        ][
            "metrics_root"
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_frames = []
    bin_frames = []

    for project_key in projects:
        print(
            f"[INFO] Evaluating "
            f"{project_key}"
        )

        (
            project_metrics,
            project_bins,
        ) = run_project(
            project_key,
            config,
        )

        metric_frames.append(
            project_metrics
        )

        bin_frames.append(
            project_bins
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    calibration_bin_data = pd.concat(
        bin_frames,
        ignore_index=True,
    )

    validate_against_baseline(
        metrics=metrics,
        baseline_root=Path(
            config[
                "inputs"
            ][
                "baseline_results_root"
            ]
        ),
    )

    metrics_path = (
        output_root
        / "pilot_metrics.csv"
    )

    bins_path = (
        output_root
        / "calibration_bins.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    calibration_bin_data.to_csv(
        bins_path,
        index=False,
    )

    summary_metrics = [
    "roc_auc",
    "average_precision",
    "ap_lift",
    "brier_score",
    "log_loss",
    "ece",
    "mean_predicted_probability",
    "calibration_bias",
    "absolute_calibration_bias",
    ]

    macro = (
        metrics
        .groupby(
            "model",
            as_index=False,
        )[
            summary_metrics
        ]
        .mean()
    )

    macro_path = (
        output_root
        / "pilot_metrics_macro.csv"
    )

    macro.to_csv(
        macro_path,
        index=False,
    )

    print()
    print(
        "[PASS] Pilot metric "
        "evaluation completed."
    )

    print()
    print(
        metrics[
            [
                "project_key",
                "model",
                "failure_prevalence",
                "roc_auc",
                "average_precision",
                "ap_lift",
                "brier_score",
                "log_loss",
                "ece",
                "mean_predicted_probability",
                "calibration_bias",
                "absolute_calibration_bias",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Macro mean across projects:"
    )

    print(
        macro.to_string(
            index=False
        )
    )

    print()
    print(
        "Metrics:",
        metrics_path,
    )

    print(
        "Calibration bins:",
        bins_path,
    )


if __name__ == "__main__":
    main()