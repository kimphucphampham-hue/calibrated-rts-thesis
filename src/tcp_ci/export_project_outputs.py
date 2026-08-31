from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/project_outputs_v1.yaml"
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


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected file artifact: {path}"
        )


def select_one_row(
    dataframe: pd.DataFrame,
    project_key: str,
    source_name: str,
) -> pd.Series:
    rows = dataframe[
        dataframe["project_key"]
        == project_key
    ]

    if len(rows) != 1:
        raise ValueError(
            f"{source_name}: expected exactly one row "
            f"for {project_key}, found {len(rows)}."
        )

    return rows.iloc[0]


def build_artifact_manifest(
    project_key: str,
    config: dict,
) -> pd.DataFrame:
    safe_name = safe_project_name(
        project_key
    )

    history_root = Path(
        config[
            "artifact_roots"
        ][
            "history"
        ]
    )

    split_root = Path(
        config[
            "artifact_roots"
        ][
            "split"
        ]
    )

    baseline_root = Path(
        config[
            "artifact_roots"
        ][
            "baseline"
        ]
    )

    artifacts = [
        (
            "history_features",
            history_root
            / safe_name
            / "test_build_history.csv",
        ),
        (
            "train_split",
            split_root
            / safe_name
            / "train.csv",
        ),
        (
            "test_split",
            split_root
            / safe_name
            / "test.csv",
        ),
        (
            "build_split",
            split_root
            / safe_name
            / "build_split.csv",
        ),
        (
            "test_predictions",
            baseline_root
            / safe_name
            / "test_predictions.csv",
        ),
    ]

    rows = []

    for (
        artifact_type,
        path,
    ) in artifacts:
        require_file(
            path
        )

        rows.append(
            {
                "project_key":
                    project_key,

                "artifact_type":
                    artifact_type,

                "path":
                    path.as_posix(),

                "size_bytes":
                    path.stat().st_size,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_project_summary(
    project_key: str,
    history_row: pd.Series,
    split_row: pd.Series,
    project_metrics: pd.DataFrame,
) -> dict:
    logistic = project_metrics[
        project_metrics["model"]
        == "logistic_regression"
    ]

    dummy = project_metrics[
        project_metrics["model"]
        == "dummy_prior"
    ]

    if len(logistic) != 1:
        raise ValueError(
            f"{project_key}: expected exactly "
            "one logistic metric row."
        )

    if len(dummy) != 1:
        raise ValueError(
            f"{project_key}: expected exactly "
            "one dummy metric row."
        )

    logistic = logistic.iloc[0]
    dummy = dummy.iloc[0]

    return {
        "project_key":
            project_key,

        "history": {
            "rows":
                int(
                    history_row["rows"]
                ),

            "unique_builds":
                int(
                    history_row[
                        "unique_builds"
                    ]
                ),

            "unique_tests":
                int(
                    history_row[
                        "unique_tests"
                    ]
                ),

            "cold_start_rows":
                int(
                    history_row[
                        "cold_start_rows"
                    ]
                ),

            "history_rows":
                int(
                    history_row[
                        "history_rows"
                    ]
                ),

            "history_rate":
                float(
                    history_row[
                        "history_rate"
                    ]
                ),
        },

        "temporal_split": {
            "total_rows":
                int(
                    split_row[
                        "total_rows"
                    ]
                ),

            "total_builds":
                int(
                    split_row[
                        "total_builds"
                    ]
                ),

            "train_rows":
                int(
                    split_row[
                        "train_rows"
                    ]
                ),

            "test_rows":
                int(
                    split_row[
                        "test_rows"
                    ]
                ),

            "train_builds":
                int(
                    split_row[
                        "train_builds"
                    ]
                ),

            "test_builds":
                int(
                    split_row[
                        "test_builds"
                    ]
                ),

            "train_failure_rate":
                float(
                    split_row[
                        "train_failure_rate"
                    ]
                ),

            "test_failure_rate":
                float(
                    split_row[
                        "test_failure_rate"
                    ]
                ),

            "train_last_build_index":
                int(
                    split_row[
                        "train_last_build_index"
                    ]
                ),

            "test_first_build_index":
                int(
                    split_row[
                        "test_first_build_index"
                    ]
                ),
        },

        "logistic_regression": {
            "roc_auc":
                float(
                    logistic["roc_auc"]
                ),

            "average_precision":
                float(
                    logistic[
                        "average_precision"
                    ]
                ),

            "ap_lift":
                float(
                    logistic["ap_lift"]
                ),

            "brier_score":
                float(
                    logistic[
                        "brier_score"
                    ]
                ),

            "log_loss":
                float(
                    logistic["log_loss"]
                ),

            "ece":
                float(
                    logistic["ece"]
                ),

            "calibration_bias":
                float(
                    logistic[
                        "calibration_bias"
                    ]
                ),

            "absolute_calibration_bias":
                float(
                    logistic[
                        "absolute_calibration_bias"
                    ]
                ),
        },

        "dummy_prior": {
            "roc_auc":
                float(
                    dummy["roc_auc"]
                ),

            "average_precision":
                float(
                    dummy[
                        "average_precision"
                    ]
                ),

            "brier_score":
                float(
                    dummy[
                        "brier_score"
                    ]
                ),

            "log_loss":
                float(
                    dummy["log_loss"]
                ),

            "ece":
                float(
                    dummy["ece"]
                ),
        },
    }


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
        in pilot_config["projects"]
    ]

    expected_project_count = int(
        config[
            "guards"
        ][
            "expected_project_count"
        ]
    )

    if len(projects) != expected_project_count:
        raise ValueError(
            f"Expected {expected_project_count} projects, "
            f"found {len(projects)}."
        )

    inputs = config[
        "inputs"
    ]

    history_summary_path = Path(
        inputs[
            "history_summary"
        ]
    )

    split_summary_path = Path(
        inputs[
            "split_summary"
        ]
    )

    baseline_summary_path = Path(
        inputs[
            "baseline_summary"
        ]
    )

    metrics_path = Path(
        inputs[
            "pilot_metrics"
        ]
    )

    bins_path = Path(
        inputs[
            "calibration_bins"
        ]
    )

    for path in [
        history_summary_path,
        split_summary_path,
        baseline_summary_path,
        metrics_path,
        bins_path,
    ]:
        require_file(path)

    history_summary = pd.read_csv(
        history_summary_path
    )

    split_summary = pd.read_csv(
        split_summary_path
    )

    baseline_summary = pd.read_csv(
        baseline_summary_path
    )

    metrics = pd.read_csv(
        metrics_path
    )

    calibration_bins = pd.read_csv(
        bins_path
    )

    # -------------------------------------------------
    # Baseline/metric identity check
    # -------------------------------------------------

    common_metrics = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ]

    left = (
        baseline_summary[
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

    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )

    output_root = Path(
        config[
            "outputs"
        ][
            "project_root"
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_config = config[
        "files"
    ]

    index_rows = []

    for project_key in projects:
        print(
            f"[INFO] Packaging {project_key}"
        )

        safe_name = safe_project_name(
            project_key
        )

        output_dir = (
            output_root
            / safe_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_row = select_one_row(
            history_summary,
            project_key,
            "history summary",
        )

        split_row = select_one_row(
            split_summary,
            project_key,
            "split summary",
        )

        project_metrics = metrics[
            metrics["project_key"]
            == project_key
        ].copy()

        expected_models = int(
            config[
                "guards"
            ][
                "expected_models_per_project"
            ]
        )

        if len(project_metrics) != expected_models:
            raise ValueError(
                f"{project_key}: expected "
                f"{expected_models} metric rows, "
                f"found {len(project_metrics)}."
            )

        project_bins = calibration_bins[
            calibration_bins[
                "project_key"
            ]
            == project_key
        ].copy()

        if project_bins.empty:
            raise ValueError(
                f"{project_key}: no "
                "calibration bins found."
            )

        summary = build_project_summary(
            project_key=project_key,
            history_row=history_row,
            split_row=split_row,
            project_metrics=project_metrics,
        )

        manifest = build_artifact_manifest(
            project_key,
            config,
        )

        summary_path = (
            output_dir
            / file_config[
                "project_summary"
            ]
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

        project_metrics.to_csv(
            output_dir
            / file_config[
                "model_metrics"
            ],
            index=False,
        )

        project_bins.to_csv(
            output_dir
            / file_config[
                "calibration_bins"
            ],
            index=False,
        )

        manifest.to_csv(
            output_dir
            / file_config[
                "artifact_manifest"
            ],
            index=False,
        )

        logistic = (
            project_metrics[
                project_metrics["model"]
                == "logistic_regression"
            ]
            .iloc[0]
        )

        index_rows.append(
            {
                "project_key":
                    project_key,

                "rows":
                    int(
                        history_row[
                            "rows"
                        ]
                    ),

                "unique_builds":
                    int(
                        history_row[
                            "unique_builds"
                        ]
                    ),

                "unique_tests":
                    int(
                        history_row[
                            "unique_tests"
                        ]
                    ),

                "test_rows":
                    int(
                        split_row[
                            "test_rows"
                        ]
                    ),

                "test_failure_rate":
                    float(
                        split_row[
                            "test_failure_rate"
                        ]
                    ),

                "logistic_roc_auc":
                    float(
                        logistic[
                            "roc_auc"
                        ]
                    ),

                "logistic_average_precision":
                    float(
                        logistic[
                            "average_precision"
                        ]
                    ),

                "logistic_brier_score":
                    float(
                        logistic[
                            "brier_score"
                        ]
                    ),

                "logistic_log_loss":
                    float(
                        logistic[
                            "log_loss"
                        ]
                    ),

                "logistic_ece":
                    float(
                        logistic[
                            "ece"
                        ]
                    ),

                "logistic_absolute_calibration_bias":
                    float(
                        logistic[
                            "absolute_calibration_bias"
                        ]
                    ),
            }
        )

    index = pd.DataFrame(
        index_rows
    )

    if len(index) != expected_project_count:
        raise AssertionError(
            "Project index row count mismatch."
        )

    index_path = (
        output_root
        / file_config[
            "project_index"
        ]
    )

    index.to_csv(
        index_path,
        index=False,
    )

    print()
    print(
        "[PASS] Per-project output "
        "packaging completed."
    )

    print()
    print(
        index.to_string(
            index=False
        )
    )

    print()
    print(
        "Project index:",
        index_path,
    )


if __name__ == "__main__":
    main()