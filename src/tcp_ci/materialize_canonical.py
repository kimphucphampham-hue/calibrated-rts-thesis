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


def materialize_project(
    project_key: str,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    interim_root = Path(
        config["outputs"]["interim_root"]
    )

    staging_path = (
        interim_root
        / safe_project_name(project_key)
        / "executions_staging.csv"
    )

    if not staging_path.exists():
        raise FileNotFoundError(
            f"Missing staging file: {staging_path}"
        )

    staging = pd.read_csv(
        staging_path,
        dtype={
            "project_id": "string",
            "build_id": "string",
            "job_id": "string",
            "test_id": "string",
        },
        low_memory=False,
    )

    required = {
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
    }

    missing = required - set(
        staging.columns
    )

    if missing:
        raise ValueError(
            f"{project_key}: missing staging columns: "
            + ", ".join(sorted(missing))
        )

    # -------------------------------------------------
    # Validate project identity
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Validate canonical key
    # -------------------------------------------------

    key = [
        "project_id",
        "build_id",
        "test_id",
    ]

    if staging[key].isna().any().any():
        raise ValueError(
            f"{project_key}: null canonical key found."
        )

    duplicate_rows = int(
        staging.duplicated(
            subset=key,
            keep=False,
        ).sum()
    )

    if duplicate_rows != 0:
        raise ValueError(
            f"{project_key}: "
            f"{duplicate_rows} duplicate "
            "canonical-key rows found."
        )

    # -------------------------------------------------
    # Validate timestamp
    # -------------------------------------------------

    build_started_at = pd.to_datetime(
        staging["build_started_at"],
        errors="coerce",
        utc=True,
    )

    if build_started_at.isna().any():
        raise ValueError(
            f"{project_key}: invalid build timestamp."
        )

    # -------------------------------------------------
    # Validate build index
    # -------------------------------------------------

    build_index = pd.to_numeric(
        staging["build_index"],
        errors="coerce",
    )

    if build_index.isna().any():
        raise ValueError(
            f"{project_key}: invalid build_index."
        )

    if not (
        build_index
        == build_index.round()
    ).all():
        raise ValueError(
            f"{project_key}: non-integer build_index."
        )

    # -------------------------------------------------
    # Validate target
    # -------------------------------------------------

    failed = pd.to_numeric(
        staging["failed"],
        errors="coerce",
    )

    if failed.isna().any():
        raise ValueError(
            f"{project_key}: invalid failed values."
        )

    observed_failed = set(
        failed.astype(int).unique()
    )

    if not observed_failed.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{project_key}: unexpected failed values: "
            f"{sorted(observed_failed)}"
        )

    # -------------------------------------------------
    # Validate duration
    # -------------------------------------------------

    duration = pd.to_numeric(
        staging["duration_raw"],
        errors="coerce",
    )

    if duration.isna().any():
        raise ValueError(
            f"{project_key}: invalid duration values."
        )

    if (duration < 0).any():
        raise ValueError(
            f"{project_key}: negative duration found."
        )

    if not (
        duration
        == duration.round()
    ).all():
        raise ValueError(
            f"{project_key}: non-integer duration found."
        )

    # TCP-CI upstream converts parsed execution duration
    # to milliseconds before creating ExecutionRecord.
    #
    # Therefore this is an identity mapping:
    #
    # exe.csv:duration
    #     ->
    # execution_time_ms
    #
    execution_time_ms = (
        duration.astype("int64")
    )

    # -------------------------------------------------
    # Materialize canonical test-build rows
    # -------------------------------------------------

    canonical = pd.DataFrame(
        {
            "project_id":
                staging["project_id"],

            "build_id":
                staging["build_id"],

            "build_started_at":
                build_started_at,

            "build_index":
                build_index.astype("int64"),

            "test_id":
                staging["test_id"],

            "execution_time_ms":
                execution_time_ms,

            "failed":
                failed.astype("int8"),

            # Retained metadata / provenance.
            "job_id":
                staging["job_id"],

            "commit_hashes":
                staging["commit_hashes"],

            "verdict_raw":
                pd.to_numeric(
                    staging["verdict_raw"],
                    errors="raise",
                ).astype("int8"),
        }
    )

    canonical = (
        canonical
        .sort_values(
            [
                "build_index",
                "test_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(canonical) != len(staging):
        raise AssertionError(
            f"{project_key}: row count changed "
            "during canonicalization."
        )

    if int(
        canonical["failed"].sum()
    ) != int(
        staging["failed"].sum()
    ):
        raise AssertionError(
            f"{project_key}: failure count changed "
            "during canonicalization."
        )

    summary = {
        "project_key": project_key,
        "staging_rows": len(staging),
        "canonical_rows": len(canonical),
        "unique_builds": int(
            canonical["build_id"].nunique()
        ),
        "unique_tests": int(
            canonical["test_id"].nunique()
        ),
        "failure_rows": int(
            canonical["failed"].sum()
        ),
        "duplicate_canonical_key_rows": int(
            canonical.duplicated(
                subset=key,
                keep=False,
            ).sum()
        ),
        "duration_missing": int(
            canonical[
                "execution_time_ms"
            ].isna().sum()
        ),
        "duration_negative": int(
            (
                canonical[
                    "execution_time_ms"
                ] < 0
            ).sum()
        ),
        "execution_time_ms_min": int(
            canonical[
                "execution_time_ms"
            ].min()
        ),
        "execution_time_ms_median": float(
            canonical[
                "execution_time_ms"
            ].median()
        ),
        "execution_time_ms_max": int(
            canonical[
                "execution_time_ms"
            ].max()
        ),
    }

    return canonical, summary


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

    duration_config = config["duration"]

    if (
        duration_config["source_unit"]
        != "milliseconds"
    ):
        raise ValueError(
            "Expected confirmed source duration "
            "unit to be milliseconds."
        )

    if (
        duration_config[
            "canonical_conversion"
        ]
        != "identity"
    ):
        raise ValueError(
            "Expected identity duration conversion."
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
        project["project_key"]
        for project in pilot_config[
            "projects"
        ]
    ]

    processed_root = Path(
        config["outputs"]["processed_root"]
    )

    diagnostics_root = Path(
        config["outputs"]["diagnostics_root"]
    )

    processed_root.mkdir(
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
            f"[INFO] Materializing {project_key}"
        )

        canonical, summary = (
            materialize_project(
                project_key,
                config,
            )
        )

        output_dir = (
            processed_root
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
            / "test_build.csv"
        )

        canonical.to_csv(
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
        / "canonical_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "[PASS] Canonical pilot "
        "materialization completed."
    )
    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Canonical summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()