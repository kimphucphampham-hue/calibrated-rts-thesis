from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/core_project_screening_v1.yaml"
)

DEFAULT_TEMPORAL_CONFIG = Path(
    "configs/temporal_split_v1.yaml"
)

DEFAULT_RQ1_CONFIG = Path(
    "configs/rq1_v1.yaml"
)

DEFAULT_PILOT_PROJECTS = Path(
    "configs/pilot_projects_v1.yaml"
)

DEFAULT_RAW_ROOT = Path(
    "data/raw/tcp_ci/main_dataset"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/core/screening"
)


REQUIRED_EXECUTION_COLUMNS = [
    "test",
    "build",
    "job",
    "verdict",
    "duration",
]

REQUIRED_BUILD_COLUMNS = [
    "id",
    "commits",
    "started_at",
]

REQUIRED_ID_MAP_COLUMNS = [
    "value",
]

REQUIRED_CHANGE_COLUMNS = [
    "Commit",
    "EntityId",
    "CommitDate",
]


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"Expected YAML mapping: {path}"
        )

    return data


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def safe_git_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()

    except Exception:
        return "UNKNOWN"


def discover_project_dirs(
    raw_root: Path,
) -> list[tuple[str, Path]]:
    exe_files = sorted(
        raw_root.rglob(
            "exe.csv"
        ),
        key=lambda path:
            path.as_posix(),
    )

    projects: list[
        tuple[str, Path]
    ] = []

    for exe_path in exe_files:
        project_dir = (
            exe_path.parent
        )

        project_key = (
            project_dir
            .relative_to(
                raw_root
            )
            .as_posix()
        )

        projects.append(
            (
                project_key,
                project_dir,
            )
        )

    return projects


def validate_csv_parse(
    path: Path,
    required_columns: list[str],
) -> int:
    row_count = 0

    reader = pd.read_csv(
        path,
        usecols=required_columns,
        chunksize=250_000,
        low_memory=False,
    )

    for chunk in reader:
        row_count += len(
            chunk
        )

    return int(
        row_count
    )


def read_executions(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=REQUIRED_EXECUTION_COLUMNS,
        dtype={
            "test": "string",
            "build": "string",
            "job": "string",
        },
        low_memory=False,
    )


def read_build_identity(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[
            "id",
            "started_at",
        ],
        dtype={
            "id": "string",
        },
        low_memory=False,
    )


def read_id_map(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=REQUIRED_ID_MAP_COLUMNS,
        dtype={
            "value": "string",
        },
        low_memory=False,
    )


def count_true(
    values: pd.Series,
) -> int:
    return int(
        values.fillna(
            False
        ).sum()
    )


def group_counts(
    executions: pd.DataFrame,
) -> tuple[int, int, int]:
    grouped = executions.groupby(
        [
            "build",
            "test",
        ],
        dropna=False,
        sort=False,
    )

    sizes = grouped.size()

    duplicate_groups = int(
        (
            sizes
            > 1
        ).sum()
    )

    jobs = grouped[
        "job"
    ].nunique(
        dropna=False
    )

    multi_job_groups = int(
        (
            jobs
            > 1
        ).sum()
    )

    verdicts = grouped[
        "verdict"
    ].nunique(
        dropna=False
    )

    conflicting_verdict_groups = int(
        (
            verdicts
            > 1
        ).sum()
    )

    return (
        duplicate_groups,
        multi_job_groups,
        conflicting_verdict_groups,
    )


def ordered_observed_builds(
    executions: pd.DataFrame,
    builds: pd.DataFrame,
) -> list[str]:
    observed = set(
        executions[
            "build"
        ]
        .dropna()
        .astype(
            "string"
        )
        .tolist()
    )

    subset = builds[
        builds[
            "id"
        ].isin(
            observed
        )
    ].copy()

    subset[
        "_timestamp"
    ] = pd.to_datetime(
        subset[
            "started_at"
        ],
        errors="coerce",
        utc=True,
    )

    numeric_id = pd.to_numeric(
        subset[
            "id"
        ],
        errors="coerce",
    )

    subset[
        "_id_is_non_numeric"
    ] = (
        numeric_id
        .isna()
        .astype(
            int
        )
    )

    subset[
        "_id_numeric"
    ] = numeric_id.fillna(
        0
    )

    subset[
        "_id_text"
    ] = (
        subset[
            "id"
        ]
        .astype(
            "string"
        )
    )

    subset = subset.sort_values(
        [
            "_timestamp",
            "_id_is_non_numeric",
            "_id_numeric",
            "_id_text",
        ],
        kind="mergesort",
    )

    return (
        subset[
            "id"
        ]
        .astype(
            "string"
        )
        .tolist()
    )


def calculate_temporal_structure(
    ordered_builds: list[str],
    temporal_config: dict[str, Any],
    rq1_config: dict[str, Any],
) -> dict[str, int]:
    total_builds = len(
        ordered_builds
    )

    train_fraction = float(
        temporal_config[
            "split"
        ][
            "train_fraction"
        ]
    )

    development_builds = int(
        math.floor(
            total_builds
            * train_fraction
        )
    )

    evaluation_builds = (
        total_builds
        - development_builds
    )

    inner_fraction = float(
        rq1_config[
            "temporal_design"
        ][
            "inner_initial"
        ][
            "model_fit_fraction_of_development"
        ]
    )

    initial_model_fit_builds = int(
        math.floor(
            development_builds
            * inner_fraction
        )
    )

    initial_calibration_builds = (
        development_builds
        - initial_model_fit_builds
    )

    return {
        "observed_builds":
            total_builds,

        "development_builds":
            development_builds,

        "evaluation_builds":
            evaluation_builds,

        "initial_model_fit_builds":
            initial_model_fit_builds,

        "initial_calibration_builds":
            initial_calibration_builds,
    }


def screen_project(
    *,
    project_key: str,
    project_dir: Path,
    config: dict[str, Any],
    temporal_config: dict[str, Any],
    rq1_config: dict[str, Any],
    pilot_projects: set[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "project_key":
            project_key,

        "pilot_anchor":
            project_key
            in pilot_projects,

        "required_files_present":
            False,

        "source_parse_ok":
            False,

        "staging_key_ok":
            False,

        "test_id_mapping_ok":
            False,

        "build_metadata_identity_ok":
            False,

        "build_linkage_ok":
            False,

        "timestamp_parse_ok":
            False,

        "canonical_key_precheck_ok":
            False,

        "target_valid":
            False,

        "execution_cost_valid":
            False,

        "chronological_history_ok":
            False,

        "model_fit_both_classes":
            False,

        "eligible_phase11":
            False,

        "exclusion_reasons":
            "",

        "parse_error":
            "",
    }

    required_names = list(
        config[
            "required_source_artifacts"
        ]
    )

    missing_files = [
        name
        for name
        in required_names
        if not (
            project_dir
            / name
        ).is_file()
    ]

    row[
        "missing_required_files"
    ] = "|".join(
        missing_files
    )

    if missing_files:
        row[
            "exclusion_reasons"
        ] = "MISSING_REQUIRED_SOURCE_ARTIFACT"

        return row

    row[
        "required_files_present"
    ] = True

    exe_path = (
        project_dir
        / "exe.csv"
    )

    builds_path = (
        project_dir
        / "builds.csv"
    )

    id_map_path = (
        project_dir
        / "id_map.csv"
    )

    change_path = (
        project_dir
        / "entity_change_history.csv"
    )

    try:
        executions = read_executions(
            exe_path
        )

        validate_csv_parse(
            builds_path,
            REQUIRED_BUILD_COLUMNS,
        )

        builds = read_build_identity(
            builds_path
        )

        id_map = read_id_map(
            id_map_path
        )

        validate_csv_parse(
            change_path,
            REQUIRED_CHANGE_COLUMNS,
        )

    except Exception as exc:
        row[
            "parse_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:500]}"
        )

        row[
            "exclusion_reasons"
        ] = "SOURCE_PARSE_ERROR"

        return row

    row[
        "source_parse_ok"
    ] = True

    row[
        "execution_rows"
    ] = int(
        len(
            executions
        )
    )

    row[
        "build_metadata_rows"
    ] = int(
        len(
            builds
        )
    )

    row[
        "unique_tests"
    ] = int(
        executions[
            "test"
        ].nunique(
            dropna=True
        )
    )

    staging_null_rows = count_true(
        executions[
            [
                "test",
                "build",
                "job",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    row[
        "staging_null_key_rows"
    ] = staging_null_rows

    row[
        "staging_key_ok"
    ] = (
        staging_null_rows
        == 0
    )

    execution_tests = set(
        executions[
            "test"
        ]
        .dropna()
        .astype(
            "string"
        )
        .tolist()
    )

    mapped_tests = set(
        id_map[
            "value"
        ]
        .dropna()
        .astype(
            "string"
        )
        .tolist()
    )

    unmapped_test_ids = (
        execution_tests
        - mapped_tests
    )

    row[
        "unmapped_test_ids"
    ] = len(
        unmapped_test_ids
    )

    row[
        "test_id_mapping_ok"
    ] = (
        len(
            unmapped_test_ids
        )
        == 0
    )

    null_build_metadata_ids = int(
        builds[
            "id"
        ]
        .isna()
        .sum()
    )

    duplicate_build_metadata_ids = int(
        builds.loc[
            builds[
                "id"
            ].duplicated(
                keep=False
            ),
            "id",
        ]
        .nunique(
            dropna=True
        )
    )

    row[
        "null_build_metadata_ids"
    ] = (
        null_build_metadata_ids
    )

    row[
        "duplicate_build_metadata_ids"
    ] = (
        duplicate_build_metadata_ids
    )

    row[
        "build_metadata_identity_ok"
    ] = (
        null_build_metadata_ids
        == 0
        and
        duplicate_build_metadata_ids
        == 0
    )

    execution_build_ids = set(
        executions[
            "build"
        ]
        .dropna()
        .astype(
            "string"
        )
        .tolist()
    )

    metadata_build_ids = set(
        builds[
            "id"
        ]
        .dropna()
        .astype(
            "string"
        )
        .tolist()
    )

    missing_build_links = (
        execution_build_ids
        - metadata_build_ids
    )

    row[
        "unlinked_execution_build_ids"
    ] = len(
        missing_build_links
    )

    row[
        "build_linkage_ok"
    ] = (
        len(
            missing_build_links
        )
        == 0
    )

    parsed_timestamps = pd.to_datetime(
        builds[
            "started_at"
        ],
        errors="coerce",
        utc=True,
    )

    timestamp_parse_failures = int(
        parsed_timestamps
        .isna()
        .sum()
    )

    row[
        "timestamp_parse_failures"
    ] = (
        timestamp_parse_failures
    )

    row[
        "timestamp_parse_ok"
    ] = (
        timestamp_parse_failures
        == 0
    )

    (
        duplicate_groups,
        multi_job_groups,
        conflicting_verdict_groups,
    ) = group_counts(
        executions
    )

    row[
        "duplicate_build_test_groups"
    ] = duplicate_groups

    row[
        "multi_job_build_test_groups"
    ] = multi_job_groups

    row[
        "conflicting_verdict_build_test_groups"
    ] = (
        conflicting_verdict_groups
    )

    row[
        "canonical_key_precheck_ok"
    ] = (
        row[
            "staging_key_ok"
        ]
        and
        duplicate_groups
        == 0
        and
        multi_job_groups
        == 0
        and
        conflicting_verdict_groups
        == 0
    )

    verdict = pd.to_numeric(
        executions[
            "verdict"
        ],
        errors="coerce",
    )

    invalid_or_missing_verdict_rows = int(
        (
            verdict.isna()
            |
            ~verdict.isin(
                config[
                    "target_validity"
                ][
                    "allowed_values"
                ]
            )
        ).sum()
    )

    row[
        "invalid_or_missing_verdict_rows"
    ] = (
        invalid_or_missing_verdict_rows
    )

    row[
        "target_valid"
    ] = (
        invalid_or_missing_verdict_rows
        == 0
    )

    duration = pd.to_numeric(
        executions[
            "duration"
        ],
        errors="coerce",
    )

    invalid_cost_rows = int(
        (
            duration.isna()
            |
            ~np.isfinite(
                duration
            )
            |
            (
                duration
                < 0
            )
        ).sum()
    )

    row[
        "invalid_execution_cost_rows"
    ] = invalid_cost_rows

    row[
        "execution_cost_valid"
    ] = (
        invalid_cost_rows
        == 0
    )

    chronology_preconditions = all(
        [
            row[
                "build_metadata_identity_ok"
            ],
            row[
                "build_linkage_ok"
            ],
            row[
                "timestamp_parse_ok"
            ],
        ]
    )

    if chronology_preconditions:
        ordered_builds = (
            ordered_observed_builds(
                executions,
                builds,
            )
        )

        temporal = (
            calculate_temporal_structure(
                ordered_builds,
                temporal_config,
                rq1_config,
            )
        )

        row.update(
            temporal
        )

        min_train_builds = int(
            temporal_config[
                "split"
            ][
                "min_train_builds"
            ]
        )

        min_test_builds = int(
            temporal_config[
                "split"
            ][
                "min_test_builds"
            ]
        )

        min_model_fit_builds = int(
            rq1_config[
                "temporal_design"
            ][
                "calibration_expansion"
            ][
                "minimum_model_fit_builds"
            ]
        )

        row[
            "chronological_history_ok"
        ] = (
            temporal[
                "development_builds"
            ]
            >= min_train_builds
            and
            temporal[
                "evaluation_builds"
            ]
            >= min_test_builds
            and
            temporal[
                "initial_model_fit_builds"
            ]
            >= min_model_fit_builds
        )

        guaranteed_model_fit_builds = (
            min_model_fit_builds
        )

        row[
            "guaranteed_model_fit_builds"
        ] = (
            guaranteed_model_fit_builds
        )

        fit_build_ids = set(
            ordered_builds[
                :
                guaranteed_model_fit_builds
            ]
        )

        fit_mask = (
            executions[
                "build"
            ].isin(
                fit_build_ids
            )
        )

        fit_verdict = verdict[
            fit_mask
        ]

        fit_failed = (
            fit_verdict
            != 0
        )

        model_fit_fail_rows = int(
            fit_failed.sum()
        )

        model_fit_pass_rows = int(
            (
                ~fit_failed
            ).sum()
        )

        row[
            "model_fit_fail_rows"
        ] = (
            model_fit_fail_rows
        )

        row[
            "model_fit_pass_rows"
        ] = (
            model_fit_pass_rows
        )

        row[
            "model_fit_both_classes"
        ] = (
            row[
                "target_valid"
            ]
            and
            model_fit_fail_rows
            > 0
            and
            model_fit_pass_rows
            > 0
        )

    else:
        row[
            "observed_builds"
        ] = int(
            executions[
                "build"
            ].nunique(
                dropna=True
            )
        )

        row[
            "development_builds"
        ] = pd.NA

        row[
            "evaluation_builds"
        ] = pd.NA

        row[
            "initial_model_fit_builds"
        ] = pd.NA

        row[
            "initial_calibration_builds"
        ] = pd.NA

        row[
            "guaranteed_model_fit_builds"
        ] = pd.NA

        row[
            "model_fit_fail_rows"
        ] = pd.NA

        row[
            "model_fit_pass_rows"
        ] = pd.NA

    reasons: list[str] = []

    checks = [
        (
            "required_files_present",
            "MISSING_REQUIRED_SOURCE_ARTIFACT",
        ),
        (
            "source_parse_ok",
            "SOURCE_PARSE_ERROR",
        ),
        (
            "staging_key_ok",
            "NULL_STAGING_KEY",
        ),
        (
            "test_id_mapping_ok",
            "UNMAPPED_TEST_ID",
        ),
        (
            "build_metadata_identity_ok",
            "AMBIGUOUS_BUILD_METADATA_ID",
        ),
        (
            "build_linkage_ok",
            "UNLINKED_EXECUTION_BUILD",
        ),
        (
            "timestamp_parse_ok",
            "INVALID_BUILD_TIMESTAMP",
        ),
        (
            "canonical_key_precheck_ok",
            "CANONICAL_KEY_AMBIGUITY",
        ),
        (
            "target_valid",
            "INVALID_TARGET",
        ),
        (
            "execution_cost_valid",
            "INVALID_EXECUTION_COST",
        ),
        (
            "chronological_history_ok",
            "INSUFFICIENT_CHRONOLOGICAL_HISTORY",
        ),
        (
            "model_fit_both_classes",
            "MODEL_FIT_SINGLE_CLASS",
        ),
    ]

    for (
        field,
        reason,
    ) in checks:
        if not bool(
            row.get(
                field,
                False,
            )
        ):
            reasons.append(
                reason
            )

    row[
        "eligible_phase11"
    ] = (
        len(
            reasons
        )
        == 0
    )

    row[
        "exclusion_reasons"
    ] = "|".join(
        reasons
    )

    return row


def validate_no_performance_fields(
    screening: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    forbidden = set(
        config[
            "forbidden_selection_fields"
        ]
    )

    overlap = (
        forbidden
        & set(
            screening.columns
        )
    )

    if overlap:
        raise AssertionError(
            "Performance fields leaked into "
            "project screening output: "
            + ", ".join(
                sorted(
                    overlap
                )
            )
        )


def run_screening(
    *,
    config_path: Path,
    temporal_config_path: Path,
    rq1_config_path: Path,
    pilot_projects_path: Path,
    raw_root: Path,
) -> pd.DataFrame:
    config = load_yaml(
        config_path
    )

    temporal_config = load_yaml(
        temporal_config_path
    )

    rq1_config = load_yaml(
        rq1_config_path
    )

    pilot_config = load_yaml(
        pilot_projects_path
    )

    pilot_projects = {
        item[
            "project_key"
        ]
        for item
        in pilot_config[
            "projects"
        ]
    }

    projects = discover_project_dirs(
        raw_root
    )

    expected_count = int(
        config[
            "project_scope"
        ][
            "discovered_projects_expected"
        ]
    )

    if len(
        projects
    ) != expected_count:
        raise AssertionError(
            "Unexpected discovered project count: "
            f"expected={expected_count}, "
            f"actual={len(projects)}"
        )

    rows = []

    for (
        index,
        (
            project_key,
            project_dir,
        ),
    ) in enumerate(
        projects,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(projects)}] "
            f"{project_key}",
            flush=True,
        )

        row = screen_project(
            project_key=
                project_key,
            project_dir=
                project_dir,
            config=
                config,
            temporal_config=
                temporal_config,
            rq1_config=
                rq1_config,
            pilot_projects=
                pilot_projects,
        )

        rows.append(
            row
        )

    screening = pd.DataFrame(
        rows
    ).sort_values(
        "project_key",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    validate_no_performance_fields(
        screening,
        config,
    )

    return screening


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=DEFAULT_TEMPORAL_CONFIG,
    )

    parser.add_argument(
        "--rq1-config",
        type=Path,
        default=DEFAULT_RQ1_CONFIG,
    )

    parser.add_argument(
        "--pilot-projects",
        type=Path,
        default=DEFAULT_PILOT_PROJECTS,
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    screening = run_screening(
        config_path=
            args.config,
        temporal_config_path=
            args.temporal_config,
        rq1_config_path=
            args.rq1_config,
        pilot_projects_path=
            args.pilot_projects,
        raw_root=
            args.raw_root,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    screening_path = (
        args.output_dir
        / "project_screening.csv"
    )

    screening.to_csv(
        screening_path,
        index=False,
    )

    config = load_yaml(
        args.config
    )

    eligible = screening[
        screening[
            "eligible_phase11"
        ]
        == True
    ]

    pilot_eligible = screening[
        (
            screening[
                "pilot_anchor"
            ]
            == True
        )
        &
        (
            screening[
                "eligible_phase11"
            ]
            == True
        )
    ]

    manifest = {
        "phase":
            "11.1B",

        "role":
            "25-project structural screening",

        "primary_final_rq1_evidence":
            False,

        "git_commit":
            safe_git_commit(),

        "discovered_projects":
            int(
                len(
                    screening
                )
            ),

        "eligible_projects":
            int(
                len(
                    eligible
                )
            ),

        "eligible_pilot_anchors":
            int(
                len(
                    pilot_eligible
                )
            ),

        "minimum_end_to_end_projects":
            int(
                config[
                    "project_scope"
                ][
                    "minimum_end_to_end_projects"
                ]
            ),

        "target_end_to_end_projects":
            int(
                config[
                    "project_scope"
                ][
                    "target_end_to_end_projects"
                ]
            ),

        "performance_fields_present":
            False,

        "model_predictions_generated":
            False,

        "model_performance_inspected":
            False,

        "input_sha256": {
            "screening_config":
                sha256_file(
                    args.config
                ),

            "temporal_config":
                sha256_file(
                    args.temporal_config
                ),

            "rq1_config":
                sha256_file(
                    args.rq1_config
                ),

            "pilot_projects":
                sha256_file(
                    args.pilot_projects
                ),
        },

        "output_sha256": {
            "project_screening":
                sha256_file(
                    screening_path
                )
        },
    }

    manifest_path = (
        args.output_dir
        / "screening_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "PHASE 11.1B STRUCTURAL SCREENING"
    )

    print(
        "projects:",
        len(
            screening
        ),
    )

    print(
        "eligible:",
        len(
            eligible
        ),
    )

    print(
        "eligible pilot anchors:",
        len(
            pilot_eligible
        ),
    )

    print(
        "performance fields present:",
        False,
    )

    print(
        "model predictions generated:",
        False,
    )

    print()
    print(
        "SCREENING COMPLETE"
    )


if __name__ == "__main__":
    main()
