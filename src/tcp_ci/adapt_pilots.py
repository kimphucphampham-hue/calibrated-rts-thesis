from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/tcp_ci_adapter_v1.yaml"
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


def normalize_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
    )


def build_metadata(
    builds: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "id",
        "commits",
        "started_at",
    }

    missing = (
        required - set(builds.columns)
    )

    if missing:
        raise ValueError(
            "builds.csv missing columns: "
            + ", ".join(sorted(missing))
        )

    result = builds[
        [
            "id",
            "commits",
            "started_at",
        ]
    ].copy()

    result["build_id"] = normalize_id(
        result["id"]
    )

    if result["build_id"].isna().any():
        raise ValueError(
            "Null build IDs found in builds.csv"
        )

    if result["build_id"].duplicated().any():
        duplicates = (
            result.loc[
                result["build_id"].duplicated(
                    keep=False
                ),
                "build_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate build IDs found: "
            + ", ".join(
                map(str, duplicates[:10])
            )
        )

    result["build_started_at"] = (
        pd.to_datetime(
            result["started_at"],
            errors="coerce",
            utc=True,
        )
    )

    if (
        result[
            "build_started_at"
        ].isna().any()
    ):
        raise ValueError(
            "Unparseable started_at values found."
        )

    result["commit_hashes"] = (
        result["commits"]
        .astype("string")
    )

    numeric_build_id = pd.to_numeric(
        result["build_id"],
        errors="coerce",
    )

    if numeric_build_id.notna().all():
        result["_build_sort"] = (
            numeric_build_id
        )
    else:
        result["_build_sort"] = (
            result["build_id"]
        )

    result = (
        result
        .sort_values(
            [
                "build_started_at",
                "_build_sort",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result["build_index"] = (
        range(
            1,
            len(result) + 1,
        )
    )

    return result[
        [
            "build_id",
            "build_started_at",
            "build_index",
            "commit_hashes",
        ]
    ]


def map_verdict(
    verdict: pd.Series,
    config: dict,
) -> pd.Series:
    numeric = pd.to_numeric(
        verdict,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            "Non-numeric or missing verdict values found."
        )

    pass_values = set(
        config[
            "verdict_mapping"
        ][
            "pass"
        ]
    )

    fail_values = set(
        config[
            "verdict_mapping"
        ][
            "fail"
        ]
    )

    allowed = (
        pass_values
        | fail_values
    )

    observed = set(
        numeric.astype(int).unique()
    )

    unexpected = (
        observed - allowed
    )

    if unexpected:
        raise ValueError(
            "Unexpected verdict values: "
            + ", ".join(
                map(
                    str,
                    sorted(unexpected),
                )
            )
        )

    mapping = {
        value: 0
        for value in pass_values
    }

    mapping.update(
        {
            value: 1
            for value in fail_values
        }
    )

    return (
        numeric
        .astype(int)
        .map(mapping)
        .astype("int8")
    )


def inspect_test_id_mapping(
    project_dir: Path,
    staging: pd.DataFrame,
) -> dict:
    id_map_path = (
        project_dir / "id_map.csv"
    )

    if not id_map_path.exists():
        return {
            "test_id_match_key_rate": None,
            "test_id_match_value_rate": None,
        }

    id_map = pd.read_csv(
        id_map_path,
        dtype="string",
        low_memory=False,
    )

    required = {
        "key",
        "value",
    }

    if not required.issubset(
        id_map.columns
    ):
        return {
            "test_id_match_key_rate": None,
            "test_id_match_value_rate": None,
        }

    tests = set(
        staging[
            "test_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if not tests:
        return {
            "test_id_match_key_rate": None,
            "test_id_match_value_rate": None,
        }

    keys = set(
        id_map["key"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = set(
        id_map["value"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return {
        "test_id_match_key_rate":
            len(tests & keys) / len(tests),

        "test_id_match_value_rate":
            len(tests & values) / len(tests),
    }


def duplicate_diagnostics(
    staging: pd.DataFrame,
    project_key: str,
) -> pd.DataFrame:
    grouped = (
        staging
        .groupby(
            [
                "build_id",
                "test_id",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            row_count=(
                "job_id",
                "size",
            ),
            job_count=(
                "job_id",
                "nunique",
            ),
            failed_unique=(
                "failed",
                "nunique",
            ),
            verdict_unique=(
                "verdict_raw",
                "nunique",
            ),
            duration_unique=(
                "duration_raw",
                "nunique",
            ),
        )
        .reset_index()
    )

    duplicates = grouped[
        grouped["row_count"] > 1
    ].copy()

    if duplicates.empty:
        return duplicates

    duplicates.insert(
        0,
        "project_key",
        project_key,
    )

    return duplicates


def adapt_project(
    project_key: str,
    config: dict,
) -> tuple[
    pd.DataFrame,
    dict,
    pd.DataFrame,
]:
    raw_root = Path(
        config["inputs"]["raw_root"]
    )

    project_dir = (
        raw_root / project_key
    )

    exe_path = (
        project_dir / "exe.csv"
    )

    builds_path = (
        project_dir / "builds.csv"
    )

    if not exe_path.exists():
        raise FileNotFoundError(
            exe_path
        )

    if not builds_path.exists():
        raise FileNotFoundError(
            builds_path
        )

    exe = pd.read_csv(
        exe_path,
        low_memory=False,
    )

    required_exe = {
        "test",
        "build",
        "job",
        "verdict",
        "duration",
    }

    missing = (
        required_exe
        - set(exe.columns)
    )

    if missing:
        raise ValueError(
            "exe.csv missing columns: "
            + ", ".join(sorted(missing))
        )

    builds = pd.read_csv(
        builds_path,
        low_memory=False,
    )

    metadata = build_metadata(
        builds
    )

    staging = pd.DataFrame(
        index=exe.index
    )

    staging["project_id"] = (
        project_key
    )

    staging["build_id"] = normalize_id(
        exe["build"]
    )

    staging["job_id"] = normalize_id(
        exe["job"]
    )

    staging["test_id"] = normalize_id(
        exe["test"]
    )

    staging["verdict_raw"] = (
        pd.to_numeric(
            exe["verdict"],
            errors="raise",
        )
        .astype(int)
    )

    staging["failed"] = map_verdict(
        exe["verdict"],
        config,
    )

    staging["duration_raw"] = (
        pd.to_numeric(
            exe["duration"],
            errors="coerce",
        )
    )
    key_columns = [
        "project_id",
        "build_id",
        "job_id",
        "test_id",
    ]

    if staging[key_columns].isna().any().any():
        null_counts = (
            staging[key_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{project_key}: null staging key values found:\n"
            f"{null_counts.to_string()}"
        )

    observed_projects = (
        staging["project_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if observed_projects != [project_key]:
        raise ValueError(
            f"{project_key}: unexpected project_id values: "
            f"{observed_projects}"
        )

    if (
        staging[
            "duration_raw"
        ].isna().any()
    ):
        raise ValueError(
            "Missing or non-numeric duration values found."
        )

    staging = staging.merge(
        metadata,
        on="build_id",
        how="left",
        validate="many_to_one",
    )

    missing_build_metadata = int(
        staging[
            "build_started_at"
        ]
        .isna()
        .sum()
    )

    if missing_build_metadata:
        raise ValueError(
            f"{missing_build_metadata} execution rows "
            "could not be linked to builds.csv."
        )

    staging = staging[
        [
            "project_id",
            "build_id",
            "build_started_at",
            "build_index",
            "job_id",
            "test_id",
            "commit_hashes",
            "verdict_raw",
            "failed",
            "duration_raw",
        ]
    ]

    staging = (
        staging
        .sort_values(
            [
                "build_index",
                "job_id",
                "test_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    duplicates = (
        duplicate_diagnostics(
            staging,
            project_key,
        )
    )

    duplicate_group_count = len(
        duplicates
    )

    if duplicates.empty:
        multi_job_group_count = 0
        conflicting_verdict_groups = 0
    else:
        multi_job_group_count = int(
            (
                duplicates[
                    "job_count"
                ] > 1
            ).sum()
        )

        conflicting_verdict_groups = int(
            (
                duplicates[
                    "failed_unique"
                ] > 1
            ).sum()
        )

    id_map_stats = (
        inspect_test_id_mapping(
            project_dir,
            staging,
        )
    )

    summary = {
        "project_key": project_key,
        "execution_rows": len(
            staging
        ),
        "unique_builds": int(
            staging[
                "build_id"
            ].nunique()
        ),
        "unique_tests": int(
            staging[
                "test_id"
            ].nunique()
        ),
        "failure_rows": int(
            staging[
                "failed"
            ].sum()
        ),
        "duplicate_test_build_groups":
            duplicate_group_count,
        "multi_job_test_build_groups":
            multi_job_group_count,
        "conflicting_verdict_groups":
            conflicting_verdict_groups,
        **id_map_stats,
    }

    return (
        staging,
        summary,
        duplicates,
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

    interim_root = Path(
        config[
            "outputs"
        ][
            "interim_root"
        ]
    )

    diagnostics_root = Path(
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
    )

    interim_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    duplicate_frames = []

    for project_key in projects:
        print(
            f"[INFO] Adapting {project_key}"
        )

        (
            staging,
            summary,
            duplicates,
        ) = adapt_project(
            project_key,
            config,
        )

        output_dir = (
            interim_root
            / safe_project_name(
                project_key
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging_path = (
            output_dir
            / "executions_staging.csv"
        )

        staging.to_csv(
            staging_path,
            index=False,
        )

        summaries.append(
            summary
        )

        if not duplicates.empty:
            duplicate_frames.append(
                duplicates
            )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        diagnostics_root
        / "adapter_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if duplicate_frames:
        duplicate_df = pd.concat(
            duplicate_frames,
            ignore_index=True,
        )
    else:
        duplicate_df = pd.DataFrame(
            columns=[
                "project_key",
                "build_id",
                "test_id",
                "row_count",
                "job_count",
                "failed_unique",
                "verdict_unique",
                "duration_unique",
            ]
        )

    duplicate_path = (
        diagnostics_root
        / "duplicate_test_builds.csv"
    )

    duplicate_df.to_csv(
        duplicate_path,
        index=False,
    )

    print()
    print(
        "[PASS] Pilot staging adapter completed."
    )
    print()

    columns = [
        "project_key",
        "execution_rows",
        "unique_builds",
        "unique_tests",
        "failure_rows",
        "duplicate_test_build_groups",
        "multi_job_test_build_groups",
        "conflicting_verdict_groups",
        "test_id_match_key_rate",
        "test_id_match_value_rate",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Summary:",
        summary_path,
    )

    print(
        "Duplicate diagnostics:",
        duplicate_path,
    )


if __name__ == "__main__":
    main()