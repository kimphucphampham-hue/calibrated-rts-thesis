from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/temporal_split_v1.yaml"
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


def build_split_table(
    dataframe: pd.DataFrame,
    train_fraction: float,
    min_train_builds: int,
    min_test_builds: int,
) -> pd.DataFrame:
    required = {
        "project_id",
        "build_id",
        "build_started_at",
        "build_index",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = dataframe.copy()

    df["build_index"] = (
        pd.to_numeric(
            df["build_index"],
            errors="raise",
        )
        .astype("int64")
    )

    df["build_started_at"] = (
        pd.to_datetime(
            df["build_started_at"],
            errors="raise",
            utc=True,
        )
    )

    # -------------------------------------------------
    # Verify build-level metadata consistency
    # -------------------------------------------------

    consistency = (
        df.groupby(
            [
                "project_id",
                "build_id",
            ],
            sort=False,
        )
        .agg(
            build_index_count=(
                "build_index",
                "nunique",
            ),
            timestamp_count=(
                "build_started_at",
                "nunique",
            ),
        )
    )

    if (
        consistency[
            "build_index_count"
        ] != 1
    ).any():
        raise ValueError(
            "A build_id maps to multiple build_index values."
        )

    if (
        consistency[
            "timestamp_count"
        ] != 1
    ).any():
        raise ValueError(
            "A build_id maps to multiple timestamps."
        )

    builds = (
        df[
            [
                "project_id",
                "build_id",
                "build_started_at",
                "build_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "project_id",
                "build_index",
                "build_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    project_values = (
        builds[
            "project_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(project_values) != 1:
        raise ValueError(
            "Expected exactly one project "
            "per split operation."
        )

    if (
        builds["build_index"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate observed build_index found."
        )

    total_builds = len(builds)

    n_train = int(
        np.floor(
            total_builds
            * train_fraction
        )
    )

    n_test = (
        total_builds
        - n_train
    )

    if n_train < min_train_builds:
        raise ValueError(
            f"Only {n_train} train builds; "
            f"minimum is {min_train_builds}."
        )

    if n_test < min_test_builds:
        raise ValueError(
            f"Only {n_test} test builds; "
            f"minimum is {min_test_builds}."
        )

    builds["split"] = "test"

    builds.loc[
        builds.index[:n_train],
        "split",
    ] = "train"

    train = builds[
        builds["split"] == "train"
    ]

    test = builds[
        builds["split"] == "test"
    ]

    if train.empty or test.empty:
        raise AssertionError(
            "Train/test split is empty."
        )

    if (
        train["build_index"].max()
        >= test["build_index"].min()
    ):
        raise AssertionError(
            "Temporal boundary violation: "
            "train is not strictly before test."
        )

    return builds


def split_project(
    dataframe: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    split_config = config["split"]

    assignments = build_split_table(
        dataframe=dataframe,
        train_fraction=float(
            split_config[
                "train_fraction"
            ]
        ),
        min_train_builds=int(
            split_config[
                "min_train_builds"
            ]
        ),
        min_test_builds=int(
            split_config[
                "min_test_builds"
            ]
        ),
    )

    merged = dataframe.merge(
        assignments[
            [
                "project_id",
                "build_id",
                "split",
            ]
        ],
        on=[
            "project_id",
            "build_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if merged["split"].isna().any():
        raise AssertionError(
            "Some rows were not assigned to a split."
        )

    train = (
        merged[
            merged["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        merged[
            merged["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if (
        len(train)
        + len(test)
        != len(dataframe)
    ):
        raise AssertionError(
            "Row count changed during split."
        )

    train_builds = set(
        train["build_id"]
        .astype(str)
        .unique()
    )

    test_builds = set(
        test["build_id"]
        .astype(str)
        .unique()
    )

    overlap = (
        train_builds
        & test_builds
    )

    if overlap:
        raise AssertionError(
            "Build leakage between train and test: "
            + ", ".join(
                sorted(overlap)[:10]
            )
        )

    train_indices = (
        train["build_index"]
        .astype(int)
    )

    test_indices = (
        test["build_index"]
        .astype(int)
    )

    if (
        train_indices.max()
        >= test_indices.min()
    ):
        raise AssertionError(
            "Train/test temporal order is invalid."
        )

    train_failure_rows = int(
        train["failed"].sum()
    )

    test_failure_rows = int(
        test["failed"].sum()
    )

    summary = {
        "project_key":
            str(
                dataframe[
                    "project_id"
                ].iloc[0]
            ),

        "total_rows":
            len(dataframe),

        "total_builds":
            len(assignments),

        "train_rows":
            len(train),

        "test_rows":
            len(test),

        "train_builds":
            len(train_builds),

        "test_builds":
            len(test_builds),

        "train_failure_rows":
            train_failure_rows,

        "test_failure_rows":
            test_failure_rows,

        "train_failure_rate":
            (
                train_failure_rows
                / len(train)
            ),

        "test_failure_rate":
            (
                test_failure_rows
                / len(test)
            ),

        "train_first_build_index":
            int(
                train_indices.min()
            ),

        "train_last_build_index":
            int(
                train_indices.max()
            ),

        "test_first_build_index":
            int(
                test_indices.min()
            ),

        "test_last_build_index":
            int(
                test_indices.max()
            ),

        "train_last_timestamp":
            str(
                pd.to_datetime(
                    train[
                        "build_started_at"
                    ],
                    utc=True,
                ).max()
            ),

        "test_first_timestamp":
            str(
                pd.to_datetime(
                    test[
                        "build_started_at"
                    ],
                    utc=True,
                ).min()
            ),

        "build_overlap_count":
            len(overlap),
    }

    return (
        train,
        test,
        assignments,
        summary,
    )


def process_project(
    project_key: str,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    history_root = Path(
        config[
            "inputs"
        ][
            "history_root"
        ]
    )

    input_path = (
        history_root
        / safe_project_name(
            project_key
        )
        / "test_build_history.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    dataframe = pd.read_csv(
        input_path,
        dtype={
            "project_id": "string",
            "build_id": "string",
            "test_id": "string",
            "job_id": "string",
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

    if project_values != [project_key]:
        raise ValueError(
            f"{project_key}: unexpected "
            f"project_id values: "
            f"{project_values}"
        )

    return split_project(
        dataframe,
        config,
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
        for item in pilot_config[
            "projects"
        ]
    ]

    split_root = Path(
        config[
            "outputs"
        ][
            "split_root"
        ]
    )

    diagnostics_root = Path(
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
    )

    split_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for project_key in projects:
        print(
            f"[INFO] Splitting {project_key}"
        )

        (
            train,
            test,
            assignments,
            summary,
        ) = process_project(
            project_key,
            config,
        )

        output_dir = (
            split_root
            / safe_project_name(
                project_key
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train.to_csv(
            output_dir / "train.csv",
            index=False,
        )

        test.to_csv(
            output_dir / "test.csv",
            index=False,
        )

        assignments.to_csv(
            output_dir / "build_split.csv",
            index=False,
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        diagnostics_root
        / "split_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "[PASS] Temporal build split completed."
    )
    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()