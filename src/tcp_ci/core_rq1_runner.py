from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from src.tcp_ci.rq1_evaluator import (
    RQ1_BIN_COLUMNS,
    RQ1_METRIC_COLUMNS,
    evaluate_rq1_predictions,
)

from src.tcp_ci.rq1_pipeline import (
    PREDICTION_ID_COLUMNS,
    RQ1_CALIBRATION_VARIANTS,
    RQ1_MODEL_NAMES,
    load_yaml,
    run_rq1_model,
    validate_rq1_integration_contract,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "core_rq1_v1.yaml"
)


def resolve_repo_path(
    value,
) -> Path:
    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        REPO_ROOT
        / path
    )


def safe_project_name(
    project_key: str,
) -> str:
    return (
        project_key
        .replace("/", "__")
        .replace("@", "__")
    )


def required_input_columns(
    feature_columns,
) -> list[str]:
    return list(
        dict.fromkeys(
            list(
                PREDICTION_ID_COLUMNS
            )
            + list(
                feature_columns
            )
        )
    )


def load_partition(
    path: Path,
    columns,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"RQ1 partition not found: "
            f"{path}"
        )

    data = pd.read_csv(
        path,
        usecols=list(
            columns
        ),
        dtype={
            "project_id":
                "string",
            "build_id":
                "string",
            "test_id":
                "string",
        },
        low_memory=False,
    )

    data[
        "build_index"
    ] = pd.to_numeric(
        data[
            "build_index"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    data[
        "failed"
    ] = pd.to_numeric(
        data[
            "failed"
        ],
        errors="raise",
    ).astype(
        "int8"
    )

    return data


def load_core_rq1_contract(
    config_path: Path,
):
    core_config = load_yaml(
        config_path
    )

    references = (
        core_config[
            "references"
        ]
    )

    rq1_config = load_yaml(
        resolve_repo_path(
            references[
                "rq1"
            ]
        )
    )

    model_config = load_yaml(
        resolve_repo_path(
            references[
                "models"
            ]
        )
    )

    calibration_config = load_yaml(
        resolve_repo_path(
            references[
                "calibration"
            ]
        )
    )

    core_features_config = (
        load_yaml(
            resolve_repo_path(
                references[
                    "core_features"
                ]
            )
        )
    )

    project_config = load_yaml(
        resolve_repo_path(
            core_config[
                "inputs"
            ][
                "project_config"
            ]
        )
    )

    feature_columns = (
        validate_rq1_integration_contract(
            rq1_config=
                rq1_config,
            model_config=
                model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )
    )

    execution = (
        core_config[
            "execution"
        ]
    )

    if (
        execution[
            "feature_set"
        ]
        != rq1_config[
            "feature_set"
        ][
            "name"
        ]
    ):
        raise ValueError(
            "CORE RQ1 feature set "
            "does not match frozen RQ1."
        )

    if list(
        execution[
            "models"
        ]
    ) != list(
        RQ1_MODEL_NAMES
    ):
        raise ValueError(
            "CORE RQ1 model list "
            "does not match frozen RQ1."
        )

    if list(
        execution[
            "calibration_variants"
        ]
    ) != list(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise ValueError(
            "CORE RQ1 calibration list "
            "does not match frozen RQ1."
        )

    projects = [
        item[
            "project_key"
        ]
        for item
        in project_config[
            "projects"
        ]
    ]

    if len(
        projects
    ) != int(
        execution[
            "project_count"
        ]
    ):
        raise ValueError(
            "CORE RQ1 project count "
            "does not match frozen "
            "project configuration."
        )

    if len(
        projects
    ) != 10:
        raise ValueError(
            "Phase 11 CORE RQ1 requires "
            "exactly 10 projects."
        )

    if (
        core_config[
            "primary_final_rq1_evidence"
        ]
        is not False
    ):
        raise ValueError(
            "Phase 11 CORE RQ1 must "
            "remain non-final evidence."
        )

    return {
        "core":
            core_config,
        "rq1":
            rq1_config,
        "models":
            model_config,
        "calibration":
            calibration_config,
        "features":
            core_features_config,
        "feature_columns":
            feature_columns,
        "projects":
            projects,
    }


def validate_project_identity(
    data: pd.DataFrame,
    project_key: str,
    partition_name: str,
) -> None:
    projects = set(
        data[
            "project_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if projects != {
        project_key
    }:
        raise ValueError(
            f"{partition_name} project "
            f"identity mismatch for "
            f"{project_key}: "
            f"{sorted(projects)}"
        )


def run_project(
    project_key: str,
    split_root: Path,
    predictions_root: Path,
    train_filename: str,
    test_filename: str,
    feature_columns,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
    n_bins: int,
):
    safe_name = (
        safe_project_name(
            project_key
        )
    )

    project_split_root = (
        split_root
        / safe_name
    )

    development = (
        load_partition(
            project_split_root
            / train_filename,
            required_input_columns(
                feature_columns
            ),
        )
    )

    evaluation = (
        load_partition(
            project_split_root
            / test_filename,
            required_input_columns(
                feature_columns
            ),
        )
    )

    validate_project_identity(
        development,
        project_key,
        "development",
    )

    validate_project_identity(
        evaluation,
        project_key,
        "evaluation",
    )

    project_prediction_root = (
        predictions_root
        / safe_name
    )

    project_prediction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_frames = []
    bin_frames = []
    diagnostics_rows = []

    for model_name in (
        RQ1_MODEL_NAMES
    ):
        print(
            f"[INFO] "
            f"{project_key} "
            f"/ {model_name}",
            flush=True,
        )

        (
            predictions,
            diagnostics,
        ) = run_rq1_model(
            development=
                development,
            evaluation=
                evaluation,
            model_name=
                model_name,
            rq1_config=
                rq1_config,
            model_config=
                model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )

        (
            metrics,
            bins,
        ) = evaluate_rq1_predictions(
            predictions=
                predictions,
            diagnostics=
                diagnostics,
            n_bins=
                n_bins,
        )

        prediction_path = (
            project_prediction_root
            / f"{model_name}.csv"
        )

        predictions.to_csv(
            prediction_path,
            index=False,
            na_rep="NA",
        )

        metric_frames.append(
            metrics
        )

        if not bins.empty:
            bin_frames.append(
                bins
            )

        diagnostics_rows.append(
            {
                "project_id":
                    project_key,
                **diagnostics,
            }
        )

        del predictions
        del metrics
        del bins
        del diagnostics

        gc.collect()

    project_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    if bin_frames:
        project_bins = pd.concat(
            bin_frames,
            ignore_index=True,
        )
    else:
        project_bins = pd.DataFrame(
            columns=
                RQ1_BIN_COLUMNS
        )

    project_diagnostics = (
        pd.DataFrame(
            diagnostics_rows
        )
    )

    del metric_frames
    del bin_frames
    del diagnostics_rows
    del development
    del evaluation

    gc.collect()

    return (
        project_metrics,
        project_bins,
        project_diagnostics,
    )


def run_core_rq1(
    config_path: Path,
):
    contract = (
        load_core_rq1_contract(
            config_path
        )
    )

    core_config = (
        contract[
            "core"
        ]
    )

    split_root = (
        resolve_repo_path(
            core_config[
                "inputs"
            ][
                "split_root"
            ]
        )
    )

    predictions_root = (
        resolve_repo_path(
            core_config[
                "outputs"
            ][
                "predictions_root"
            ]
        )
    )

    predictions_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_filename = (
        core_config[
            "inputs"
        ][
            "train_filename"
        ]
    )

    test_filename = (
        core_config[
            "inputs"
        ][
            "test_filename"
        ]
    )

    n_bins = int(
        contract[
            "calibration"
        ][
            "metrics"
        ][
            "ece"
        ][
            "n_bins"
        ]
    )

    all_metrics = []
    all_bins = []
    all_diagnostics = []

    for project_key in (
        contract[
            "projects"
        ]
    ):
        print(
            f"[INFO] RQ1 project: "
            f"{project_key}",
            flush=True,
        )

        (
            project_metrics,
            project_bins,
            project_diagnostics,
        ) = run_project(
            project_key=
                project_key,
            split_root=
                split_root,
            predictions_root=
                predictions_root,
            train_filename=
                train_filename,
            test_filename=
                test_filename,
            feature_columns=
                contract[
                    "feature_columns"
                ],
            rq1_config=
                contract[
                    "rq1"
                ],
            model_config=
                contract[
                    "models"
                ],
            calibration_config=
                contract[
                    "calibration"
                ],
            core_features_config=
                contract[
                    "features"
                ],
            n_bins=
                n_bins,
        )

        all_metrics.append(
            project_metrics
        )

        if not project_bins.empty:
            all_bins.append(
                project_bins
            )

        all_diagnostics.append(
            project_diagnostics
        )

        del project_metrics
        del project_bins
        del project_diagnostics

        gc.collect()

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    if all_bins:
        bins = pd.concat(
            all_bins,
            ignore_index=True,
        )
    else:
        bins = pd.DataFrame(
            columns=
                RQ1_BIN_COLUMNS
        )

    diagnostics = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    expected_conditions = int(
        core_config[
            "execution"
        ][
            "maximum_conditions"
        ]
    )

    if len(
        metrics
    ) != expected_conditions:
        raise AssertionError(
            "Unexpected CORE RQ1 metric "
            f"condition count: "
            f"{len(metrics)} "
            f"!= {expected_conditions}"
        )

    if list(
        metrics.columns
    ) != list(
        RQ1_METRIC_COLUMNS
    ):
        raise AssertionError(
            "CORE RQ1 metric schema "
            "does not match evaluator."
        )

    condition_duplicates = (
        metrics[
            [
                "project_id",
                "model",
                "calibration_variant",
            ]
        ]
        .duplicated()
        .sum()
    )

    if int(
        condition_duplicates
    ) != 0:
        raise AssertionError(
            "Duplicate CORE RQ1 "
            "metric conditions found."
        )

    expected_diagnostics = (
        len(
            contract[
                "projects"
            ]
        )
        * len(
            RQ1_MODEL_NAMES
        )
    )

    if len(
        diagnostics
    ) != expected_diagnostics:
        raise AssertionError(
            "Unexpected CORE RQ1 "
            "diagnostic row count."
        )

    metrics_path = (
        resolve_repo_path(
            core_config[
                "outputs"
            ][
                "metrics"
            ]
        )
    )

    bins_path = (
        resolve_repo_path(
            core_config[
                "outputs"
            ][
                "calibration_bins"
            ]
        )
    )

    diagnostics_path = (
        resolve_repo_path(
            core_config[
                "outputs"
            ][
                "diagnostics"
            ]
        )
    )

    for path in [
        metrics_path,
        bins_path,
        diagnostics_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    metrics.to_csv(
        metrics_path,
        index=False,
        na_rep="NA",
    )

    bins.to_csv(
        bins_path,
        index=False,
        na_rep="NA",
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
        na_rep="NA",
    )

    print()
    print(
        "[PASS] CORE RQ1 execution "
        "completed.",
        flush=True,
    )

    print(
        f"Projects: "
        f"{len(contract['projects'])}",
        flush=True,
    )

    print(
        f"Metric conditions: "
        f"{len(metrics)}",
        flush=True,
    )

    print(
        f"Model diagnostics: "
        f"{len(diagnostics)}",
        flush=True,
    )

    print(
        f"Metrics: {metrics_path}",
        flush=True,
    )

    print(
        f"Calibration bins: "
        f"{bins_path}",
        flush=True,
    )

    print(
        f"Diagnostics: "
        f"{diagnostics_path}",
        flush=True,
    )

    return (
        metrics,
        bins,
        diagnostics,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Phase 11 "
            "ten-project CORE RQ1 "
            "implementation validation."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "CORE RQ1 execution "
            "configuration."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = (
        args.config
        if args.config.is_absolute()
        else resolve_repo_path(
            args.config
        )
    )

    run_core_rq1(
        config_path
    )


if __name__ == "__main__":
    main()
