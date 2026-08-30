from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/history_features_v1.yaml"
)


FEATURE_COLUMNS = [
    "failure_rate_history",
    "recent_failure",
    "consecutive_failures",
    "consecutive_passes",
    "test_age_builds",
    "time_since_last_failure_builds",
    "execution_time_mean_history_ms",
]


SUPPORT_COLUMNS = [
    "history_available",
    "prior_execution_count",
    "prior_failure_count",
]


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


def compute_history_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "project_id",
        "build_id",
        "build_started_at",
        "build_index",
        "test_id",
        "execution_time_ms",
        "failed",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Missing canonical columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = dataframe.copy()

    result["build_index"] = (
        pd.to_numeric(
            result["build_index"],
            errors="raise",
        )
        .astype("int64")
    )

    result["failed"] = (
        pd.to_numeric(
            result["failed"],
            errors="raise",
        )
        .astype("int8")
    )

    result["execution_time_ms"] = (
        pd.to_numeric(
            result[
                "execution_time_ms"
            ],
            errors="raise",
        )
        .astype("int64")
    )

    result["build_started_at"] = (
        pd.to_datetime(
            result[
                "build_started_at"
            ],
            utc=True,
            errors="raise",
        )
    )

    result = (
        result
        .sort_values(
            [
                "build_index",
                "test_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    key = [
        "project_id",
        "build_id",
        "test_id",
    ]

    if (
        result[key]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Null canonical key found."
        )

    if result.duplicated(
        subset=key
    ).any():
        raise ValueError(
            "Duplicate canonical key found."
        )

    if not set(
        result["failed"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "Unexpected failed value."
        )

    if (
        result[
            "execution_time_ms"
        ] < 0
    ).any():
        raise ValueError(
            "Negative execution time found."
        )

    # -------------------------------------------------
    # Prior execution counts
    # -------------------------------------------------

    result[
        "prior_execution_count"
    ] = (
        result
        .groupby(
            "test_id",
            sort=False,
        )
        .cumcount()
        .astype("int64")
    )

    result[
        "history_available"
    ] = (
        result[
            "prior_execution_count"
        ] > 0
    ).astype("int8")

    # -------------------------------------------------
    # Prior failure count
    #
    # Current failed value is subtracted so the feature
    # contains only outcomes strictly before this row.
    # -------------------------------------------------

    cumulative_failures = (
        result
        .groupby(
            "test_id",
            sort=False,
        )["failed"]
        .cumsum()
    )

    result[
        "prior_failure_count"
    ] = (
        cumulative_failures
        - result["failed"]
    ).astype("int64")

    denominator = (
        result[
            "prior_execution_count"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    result[
        "failure_rate_history"
    ] = (
        result[
            "prior_failure_count"
        ]
        / denominator
    )

    # -------------------------------------------------
    # Most recent prior verdict
    # -------------------------------------------------

    result[
        "recent_failure"
    ] = (
        result
        .groupby(
            "test_id",
            sort=False,
        )["failed"]
        .shift(1)
    )

    # -------------------------------------------------
    # Test age
    #
    # cummin is causal: at each row it only reflects
    # the first build observed up through that row.
    # -------------------------------------------------

    first_seen_build = (
        result
        .groupby(
            "test_id",
            sort=False,
        )["build_index"]
        .cummin()
    )

    result[
        "test_age_builds"
    ] = (
        result["build_index"]
        - first_seen_build
    ).astype("int64")

    # -------------------------------------------------
    # Most recent prior failure build
    # -------------------------------------------------

    result[
        "_failure_build"
    ] = (
        result[
            "build_index"
        ]
        .where(
            result[
                "failed"
            ].eq(1)
        )
    )

    previous_failure_candidate = (
        result
        .groupby(
            "test_id",
            sort=False,
        )["_failure_build"]
        .shift(1)
    )

    last_failure_before = (
        previous_failure_candidate
        .groupby(
            result["test_id"],
            sort=False,
        )
        .ffill()
    )

    result[
        "time_since_last_failure_builds"
    ] = (
        result["build_index"]
        - last_failure_before
    )

    # -------------------------------------------------
    # Historical execution-time mean
    #
    # Current duration is removed from cumulative sum,
    # so only durations strictly before this row remain.
    # -------------------------------------------------

    cumulative_duration = (
        result
        .groupby(
            "test_id",
            sort=False,
        )[
            "execution_time_ms"
        ]
        .cumsum()
    )

    prior_duration_sum = (
        cumulative_duration
        - result[
            "execution_time_ms"
        ]
    )

    result[
        "execution_time_mean_history_ms"
    ] = (
        prior_duration_sum
        / denominator
    )

    # -------------------------------------------------
    # Consecutive failure/pass streak immediately
    # before the current execution.
    # -------------------------------------------------

    previous_outcome = (
        result[
            "recent_failure"
        ]
    )

    run_start = (
        previous_outcome.isna()
        |
        result["failed"].ne(
            previous_outcome
        )
    )

    result["_run_id"] = (
        run_start
        .groupby(
            result["test_id"],
            sort=False,
        )
        .cumsum()
        .astype("int64")
    )

    result["_run_position"] = (
        result
        .groupby(
            [
                "test_id",
                "_run_id",
            ],
            sort=False,
        )
        .cumcount()
        .add(1)
        .astype("int64")
    )

    previous_run_position = (
        result
        .groupby(
            "test_id",
            sort=False,
        )[
            "_run_position"
        ]
        .shift(1)
        .fillna(0)
    )

    previous_is_failure = (
        previous_outcome
        .eq(1)
        .fillna(False)
    )

    previous_is_pass = (
        previous_outcome
        .eq(0)
        .fillna(False)
    )

    result[
        "consecutive_failures"
    ] = np.where(
        previous_is_failure,
        previous_run_position,
        0,
    ).astype("int64")

    result[
        "consecutive_passes"
    ] = np.where(
        previous_is_pass,
        previous_run_position,
        0,
    ).astype("int64")

    # -------------------------------------------------
    # Internal causal / cold-start validations
    # -------------------------------------------------

    cold_start = (
        result[
            "prior_execution_count"
        ] == 0
    )

    with_history = ~cold_start

    if not (
        result.loc[
            cold_start,
            "failure_rate_history",
        ].isna().all()
    ):
        raise AssertionError(
            "Cold-start failure rate must be missing."
        )

    if not (
        result.loc[
            cold_start,
            "recent_failure",
        ].isna().all()
    ):
        raise AssertionError(
            "Cold-start recent_failure must be missing."
        )

    if not (
        result.loc[
            cold_start,
            "execution_time_mean_history_ms",
        ].isna().all()
    ):
        raise AssertionError(
            "Cold-start duration mean must be missing."
        )

    if not (
        result.loc[
            cold_start,
            "time_since_last_failure_builds",
        ].isna().all()
    ):
        raise AssertionError(
            "Cold-start last-failure age must be missing."
        )

    if not (
        result.loc[
            cold_start,
            "test_age_builds",
        ] == 0
    ).all():
        raise AssertionError(
            "Cold-start test age must be zero."
        )

    if not (
        result.loc[
            cold_start,
            "consecutive_failures",
        ] == 0
    ).all():
        raise AssertionError(
            "Cold-start failure streak must be zero."
        )

    if not (
        result.loc[
            cold_start,
            "consecutive_passes",
        ] == 0
    ).all():
        raise AssertionError(
            "Cold-start pass streak must be zero."
        )

    required_with_history = [
        "failure_rate_history",
        "recent_failure",
        "execution_time_mean_history_ms",
    ]

    if (
        result.loc[
            with_history,
            required_with_history,
        ]
        .isna()
        .any()
        .any()
    ):
        raise AssertionError(
            "History feature unexpectedly missing."
        )

    if not (
        result[
            "prior_failure_count"
        ]
        <= result[
            "prior_execution_count"
        ]
    ).all():
        raise AssertionError(
            "Prior failures exceed prior executions."
        )

    observed_rates = (
        result[
            "failure_rate_history"
        ]
        .dropna()
    )

    if not observed_rates.between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Invalid historical failure rate."
        )

    observed_failure_ages = (
        result[
            "time_since_last_failure_builds"
        ]
        .dropna()
    )

    if not (
        observed_failure_ages > 0
    ).all():
        raise AssertionError(
            "Last-failure distance must be positive."
        )

    if not (
        result[
            "execution_time_mean_history_ms"
        ]
        .dropna()
        >= 0
    ).all():
        raise AssertionError(
            "Historical duration mean is negative."
        )

    result.drop(
        columns=[
            "_failure_build",
            "_run_id",
            "_run_position",
        ],
        inplace=True,
    )

    return result


def build_project(
    project_key: str,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    canonical_root = Path(
        config[
            "inputs"
        ][
            "canonical_root"
        ]
    )

    input_path = (
        canonical_root
        / safe_project_name(
            project_key
        )
        / "test_build.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    canonical = pd.read_csv(
        input_path,
        dtype={
            "project_id": "string",
            "build_id": "string",
            "test_id": "string",
            "job_id": "string",
        },
        low_memory=False,
    )

    featured = (
        compute_history_features(
            canonical
        )
    )

    cold_start_rows = int(
        (
            featured[
                "history_available"
            ] == 0
        ).sum()
    )

    history_rows = int(
        (
            featured[
                "history_available"
            ] == 1
        ).sum()
    )

    summary = {
        "project_key": project_key,
        "rows": len(featured),
        "unique_builds": int(
            featured[
                "build_id"
            ].nunique()
        ),
        "unique_tests": int(
            featured[
                "test_id"
            ].nunique()
        ),
        "cold_start_rows":
            cold_start_rows,
        "history_rows":
            history_rows,
        "history_rate":
            history_rows
            / len(featured),
        "rows_with_prior_failure": int(
            featured[
                "time_since_last_failure_builds"
            ]
            .notna()
            .sum()
        ),
        "max_prior_execution_count": int(
            featured[
                "prior_execution_count"
            ].max()
        ),
    }

    return featured, summary


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

    configured_features = (
        config[
            "feature_columns"
        ]
    )

    if configured_features != FEATURE_COLUMNS:
        raise ValueError(
            "Config feature list does not match "
            "implementation feature list."
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

    feature_root = Path(
        config[
            "outputs"
        ][
            "feature_root"
        ]
    )

    diagnostics_root = Path(
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
    )

    feature_root.mkdir(
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
            f"[INFO] Building history features "
            f"for {project_key}"
        )

        featured, summary = (
            build_project(
                project_key,
                config,
            )
        )

        output_dir = (
            feature_root
            / safe_project_name(
                project_key
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "test_build_history.csv"
        )

        featured.to_csv(
            output_path,
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
        / "history_feature_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "[PASS] Historical feature "
        "construction completed."
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