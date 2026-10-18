from __future__ import annotations

import argparse
import gc
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
    "changed_entity_count",
    "recent_failure_density_10",
    "execution_time_trend_10",
]


HISTORY_WINDOW = 10


SUPPORT_COLUMNS = [
    "history_available",
    "prior_execution_count",
    "prior_failure_count",
]

HISTORY_GROUP_KEYS = [
    "project_id",
    "test_id",
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
                "project_id",
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
            HISTORY_GROUP_KEYS,
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
            HISTORY_GROUP_KEYS,
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
            HISTORY_GROUP_KEYS,
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
            HISTORY_GROUP_KEYS,
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
            HISTORY_GROUP_KEYS,
            sort=False,
        )["_failure_build"]
        .shift(1)
    )

    last_failure_before = (
        previous_failure_candidate
        .groupby(
            [
                result["project_id"],
                result["test_id"],
            ],
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
            HISTORY_GROUP_KEYS,
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
    # Recent failure density over the previous
    # HISTORY_WINDOW observed executions.
    #
    # shift(1) removes the current target observation.
    # -------------------------------------------------

    previous_failed = (
        result
        .groupby(
            HISTORY_GROUP_KEYS,
            sort=False,
        )["failed"]
        .shift(1)
    )

    history_groupers = [
        result["project_id"],
        result["test_id"],
    ]

    result[
        "recent_failure_density_10"
    ] = (
        previous_failed
        .groupby(
            history_groupers,
            sort=False,
        )
        .rolling(
            window=HISTORY_WINDOW,
            min_periods=HISTORY_WINDOW,
        )
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True,
        )
        .sort_index()
    )

    # -------------------------------------------------
    # Execution-time trend over the previous
    # HISTORY_WINDOW observed executions.
    #
    # OLS slope:
    #
    #   (n * sum(xy) - sum(x) * sum(y))
    #   ---------------------------------
    #   (n * sum(x^2) - sum(x)^2)
    #
    # x = build_index
    # y = execution_time_ms
    #
    # Current observation is excluded by shift(1).
    # -------------------------------------------------

    previous_build_index = (
        result
        .groupby(
            HISTORY_GROUP_KEYS,
            sort=False,
        )["build_index"]
        .shift(1)
        .astype(float)
    )

    previous_duration = (
        result
        .groupby(
            HISTORY_GROUP_KEYS,
            sort=False,
        )["execution_time_ms"]
        .shift(1)
        .astype(float)
    )

    def rolling_sum(
        series: pd.Series,
    ) -> pd.Series:
        return (
            series
            .groupby(
                history_groupers,
                sort=False,
            )
            .rolling(
                window=HISTORY_WINDOW,
                min_periods=HISTORY_WINDOW,
            )
            .sum()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
            .sort_index()
        )

    sum_x = rolling_sum(
        previous_build_index
    )

    sum_y = rolling_sum(
        previous_duration
    )

    sum_x2 = rolling_sum(
        previous_build_index
        * previous_build_index
    )

    sum_xy = rolling_sum(
        previous_build_index
        * previous_duration
    )

    n = float(HISTORY_WINDOW)

    slope_denominator = (
        n * sum_x2
        - sum_x * sum_x
    )

    slope_numerator = (
        n * sum_xy
        - sum_x * sum_y
    )

    result[
        "execution_time_trend_10"
    ] = (
        slope_numerator
        / slope_denominator.replace(
            0.0,
            np.nan,
        )
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
            [
                result["project_id"],
                result["test_id"],
            ],
            sort=False,
        )
        .cumsum()
        .astype("int64")
    )

    result["_run_position"] = (
        result
        .groupby(
            [
                "project_id",
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
            HISTORY_GROUP_KEYS,
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

    insufficient_window = (
        result[
            "prior_execution_count"
        ] < HISTORY_WINDOW
    )

    sufficient_window = (
        result[
            "prior_execution_count"
        ] >= HISTORY_WINDOW
    )

    if not (
        result.loc[
            insufficient_window,
            "recent_failure_density_10",
        ]
        .isna()
        .all()
    ):
        raise AssertionError(
            "Recent failure density must be "
            "missing before 10 prior executions."
        )

    if not (
        result.loc[
            insufficient_window,
            "execution_time_trend_10",
        ]
        .isna()
        .all()
    ):
        raise AssertionError(
            "Execution-time trend must be "
            "missing before 10 prior executions."
        )

    if (
        result.loc[
            sufficient_window,
            "recent_failure_density_10",
        ]
        .isna()
        .any()
    ):
        raise AssertionError(
            "Recent failure density unexpectedly missing."
        )

    observed_density = (
        result[
            "recent_failure_density_10"
        ]
        .dropna()
    )

    if not observed_density.between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Recent failure density outside [0, 1]."
        )

    if (
        result.loc[
            sufficient_window,
            "execution_time_trend_10",
        ]
        .isna()
        .any()
    ):
        raise AssertionError(
            "Execution-time trend unexpectedly missing."
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

def parse_commit_tokens(
    value,
) -> list[str]:
    if pd.isna(value):
        return []

    return [
        token.strip()
        for token in str(value).split("#")
        if token.strip()
    ]


def compute_changed_entity_count(
    canonical: pd.DataFrame,
    raw_project_dir: Path,
) -> pd.DataFrame:
    builds_path = (
        raw_project_dir
        / "builds.csv"
    )

    changes_path = (
        raw_project_dir
        / "entity_change_history.csv"
    )

    if not builds_path.exists():
        raise FileNotFoundError(
            builds_path
        )

    if not changes_path.exists():
        raise FileNotFoundError(
            changes_path
        )

    builds = pd.read_csv(
        builds_path,
        dtype={
            "id": "string",
            "commits": "string",
        },
        low_memory=False,
    )

    changes = pd.read_csv(
        changes_path,
        dtype={
            "Commit": "string",
        },
        low_memory=False,
    )

    required_build_columns = {
        "id",
        "commits",
        "started_at",
    }

    required_change_columns = {
        "Commit",
        "CommitDate",
        "EntityId",
    }

    missing_build_columns = (
        required_build_columns
        - set(builds.columns)
    )

    if missing_build_columns:
        raise ValueError(
            "builds.csv missing required columns: "
            + ", ".join(
                sorted(
                    missing_build_columns
                )
            )
        )

    missing_change_columns = (
        required_change_columns
        - set(changes.columns)
    )

    if missing_change_columns:
        raise ValueError(
            "entity_change_history.csv "
            "missing required columns: "
            + ", ".join(
                sorted(
                    missing_change_columns
                )
            )
        )

    builds["id"] = (
        builds["id"]
        .astype("string")
    )

    builds["started_at"] = (
        pd.to_datetime(
            builds["started_at"],
            utc=True,
            errors="raise",
        )
    )

    changes["Commit"] = (
        changes["Commit"]
        .astype("string")
    )

    changes["CommitDate"] = (
        pd.to_datetime(
            changes["CommitDate"],
            utc=True,
            errors="raise",
        )
    )

    canonical_builds = (
        canonical[
            [
                "build_id",
                "build_started_at",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    canonical_builds[
        "build_id"
    ] = (
        canonical_builds[
            "build_id"
        ]
        .astype("string")
    )

    canonical_builds[
        "build_started_at"
    ] = pd.to_datetime(
        canonical_builds[
            "build_started_at"
        ],
        utc=True,
        errors="raise",
    )

    raw_builds = (
        builds[
            [
                "id",
                "commits",
                "started_at",
            ]
        ]
        .rename(
            columns={
                "id": "build_id",
                "started_at":
                    "raw_build_started_at",
            }
        )
    )

    build_context = (
        canonical_builds
        .merge(
            raw_builds,
            on="build_id",
            how="left",
            validate="one_to_one",
        )
    )

    result_rows = []

    changes_by_commit = {
        commit: frame
        for commit, frame in changes.groupby(
            "Commit",
            sort=False,
        )
    }

    for row in build_context.itertuples(
        index=False
    ):
        commits = parse_commit_tokens(
            row.commits
        )

        feature_value = np.nan

        if commits:
            matched_frames = []
            all_commits_matched = True

            for commit in commits:
                frame = (
                    changes_by_commit.get(
                        commit
                    )
                )

                if frame is None:
                    all_commits_matched = False
                    break

                matched_frames.append(
                    frame
                )

            if all_commits_matched:
                matched = pd.concat(
                    matched_frames,
                    ignore_index=True,
                )

                has_future_change = (
                    matched[
                        "CommitDate"
                    ]
                    > row.build_started_at
                ).any()

                if not has_future_change:
                    feature_value = float(
                        matched[
                            "EntityId"
                        ]
                        .nunique(
                            dropna=True
                        )
                    )

        result_rows.append(
            {
                "build_id":
                    str(row.build_id),
                "changed_entity_count":
                    feature_value,
            }
        )

    return pd.DataFrame(
        result_rows
    )

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

    data_sources = load_yaml(
        Path(
            config[
                "inputs"
            ][
                "data_sources_config"
            ]
        )
    )

    raw_project_root = Path(
        data_sources[
            "tcp_ci"
        ][
            "project_root"
        ]
    )

    raw_project_name = (
        project_key.removeprefix(
            "datasets/"
        )
    )

    raw_project_dir = (
        raw_project_root
        / raw_project_name
    )

    featured = (
        compute_history_features(
            canonical
        )
    )

    change_features = (
        compute_changed_entity_count(
            canonical=canonical,
            raw_project_dir=raw_project_dir,
        )
    )

    featured = (
        featured
        .merge(
            change_features,
            on="build_id",
            how="left",
            validate="many_to_one",
        )
    )

    if len(featured) != len(canonical):
        raise AssertionError(
            "Row count changed while adding "
            "changed_entity_count."
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

        # Release the potentially large per-project feature
        # dataframe before processing the next project.
        #
        # This is a resource-management step only. Generated
        # feature values and experimental semantics are unchanged.
        del featured
        gc.collect()

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