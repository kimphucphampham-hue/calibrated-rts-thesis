from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from src.tcp_ci.probability_metrics import (
        calibration_bins,
        evaluate_model,
        expected_calibration_error,
        validate_probability,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise

    from probability_metrics import (
        calibration_bins,
        evaluate_model,
        expected_calibration_error,
        validate_probability,
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
