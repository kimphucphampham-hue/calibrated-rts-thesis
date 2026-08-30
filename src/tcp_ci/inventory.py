"""Inventory the downloaded TCP-CI dataset.

This script performs read-only inspection of the raw TCP-CI
artifact.

It does NOT:
- modify raw files;
- normalize data;
- perform feature engineering;
- select pilot projects;
- train any model.

Run from the repository root:

    python src/tcp_ci/inventory.py

Outputs are written to:

    results/pilot/inventory/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_ROOT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "tcp_ci"
    / "main_dataset"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "pilot"
    / "inventory"
)


# ============================================================
# ARTIFACTS DOCUMENTED FOR TCP-CI
# ============================================================

EXPECTED_FILES = [
    "dataset.csv",
    "exe.csv",
    "builds.csv",
    "entity_change_history.csv",
    "id_map.csv",
    "contributors.csv",
]

ANCHOR_FILES = {
    "dataset.csv",
    "exe.csv",
    "builds.csv",
}


# ============================================================
# COLUMN CANDIDATES
#
# These are detection hints only.
#
# Finding one of these names does NOT freeze the canonical
# schema. Actual mappings are reviewed after inventory.
# ============================================================

TEST_ID_CANDIDATES = [
    "test_id",
    "test_name",
    "testName",
    "test",
    "test_case",
    "test_case_id",
]

BUILD_ID_CANDIDATES = [
    "build_id",
    "build",
    "travisBuildId",
    "tr_build_id",
]

TEST_RESULT_CANDIDATES = [
    "verdict",
    "test_result",
]

DURATION_CANDIDATES = [
    "duration",
]

BUILD_TIMESTAMP_CANDIDATES = [
    "gh_build_started_at",
    "build_started_at",
    "started_at",
    "timestamp",
]


# ============================================================
# GENERIC HELPERS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the downloaded TCP-CI dataset "
            "without modifying raw artifacts."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory of the extracted TCP-CI dataset.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where inventory outputs are written.",
    )

    return parser.parse_args()


def first_matching_column(
    columns: List[str],
    candidates: List[str],
) -> Optional[str]:
    """Return the first candidate found in columns."""

    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Fall back to case-insensitive matching.
    lowercase_lookup = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        match = lowercase_lookup.get(
            candidate.lower()
        )

        if match is not None:
            return match

    return None


def read_csv_header(path: Path) -> List[str]:
    """Read only the CSV header."""

    try:
        frame = pd.read_csv(
            path,
            nrows=0,
        )

        return frame.columns.tolist()

    except UnicodeDecodeError:
        frame = pd.read_csv(
            path,
            nrows=0,
            encoding="latin-1",
        )

        return frame.columns.tolist()


def count_csv_rows(path: Path) -> int:
    """Count logical CSV rows without loading the whole file."""

    # Some TCP-CI CSV fields, especially commit-history fields,
    # can be larger than Python's default csv field-size limit.
    field_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10

    encodings = [
        "utf-8-sig",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            with path.open(
                "r",
                encoding=encoding,
                errors="strict",
                newline="",
            ) as file:
                reader = csv.reader(file)

                # Skip header.
                next(reader, None)

                return sum(
                    1
                    for _ in reader
                )

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(
        f"Could not decode CSV file: {path}"
    ) from last_error


def file_size_mb(path: Path) -> float:
    """Return file size in MiB."""

    return round(
        path.stat().st_size
        / (1024 * 1024),
        3,
    )


def schema_signature(
    columns: List[str],
) -> str:
    """Create a stable short hash for a column schema."""

    value = "\n".join(columns)

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:12]


# ============================================================
# PROJECT DISCOVERY
# ============================================================

def discover_project_dirs(
    raw_root: Path,
) -> List[Path]:
    """Discover directories containing key TCP-CI artifacts.

    The function deliberately calls these directories
    'project directories' only as an inventory convention.

    The actual local hierarchy is preserved in project_key.
    """

    project_dirs = set()

    for filename in ANCHOR_FILES:
        for path in raw_root.rglob(filename):
            project_dirs.add(
                path.parent
            )

    return sorted(
        project_dirs,
        key=lambda path: (
            path.relative_to(
                raw_root
            ).as_posix()
        ),
    )


def project_key(
    project_dir: Path,
    raw_root: Path,
) -> str:
    """Return stable relative project/artifact path."""

    relative = project_dir.relative_to(
        raw_root
    )

    value = relative.as_posix()

    if value == ".":
        return "__root__"

    return value


# ============================================================
# FILE INVENTORY
# ============================================================

def inspect_expected_file(
    project: str,
    path: Path,
) -> Dict[str, object]:
    """Inspect one known CSV artifact."""

    try:
        columns = read_csv_header(
            path
        )

        row_count = count_csv_rows(
            path
        )

        return {
            "project_key": project,
            "artifact": path.name,
            "path": str(
                path.relative_to(
                    REPO_ROOT
                )
            ),
            "size_mb": file_size_mb(
                path
            ),
            "row_count": row_count,
            "column_count": len(
                columns
            ),
            "schema_signature": (
                schema_signature(
                    columns
                )
            ),
            "columns": json.dumps(
                columns,
                ensure_ascii=False,
            ),
            "read_status": "OK",
            "error": "",
        }

    except Exception as error:
        return {
            "project_key": project,
            "artifact": path.name,
            "path": str(
                path.relative_to(
                    REPO_ROOT
                )
            ),
            "size_mb": file_size_mb(
                path
            ),
            "row_count": None,
            "column_count": None,
            "schema_signature": "",
            "columns": "[]",
            "read_status": "ERROR",
            "error": repr(error),
        }


# ============================================================
# EXE.CSV INSPECTION
# ============================================================

def inspect_exe_file(
    path: Path,
) -> Dict[str, object]:
    """Collect conservative statistics from exe.csv.

    Statistics are computed only when recognizable columns
    actually exist.

    Column detection is evidence gathering, not canonical
    schema confirmation.
    """

    result: Dict[str, object] = {
        "exe_test_id_column": "",
        "exe_build_id_column": "",
        "exe_test_result_column": "",
        "exe_duration_column": "",
        "unique_tests": None,
        "unique_builds_in_exe": None,
        "pass_rows": None,
        "fail_rows": None,
        "other_result_rows": None,
        "failure_rate": None,
        "duration_non_null": None,
        "duration_missing": None,
        "duration_min": None,
        "duration_median": None,
        "duration_max": None,
    }

    try:
        columns = read_csv_header(
            path
        )

        test_id_column = (
            first_matching_column(
                columns,
                TEST_ID_CANDIDATES,
            )
        )

        build_id_column = (
            first_matching_column(
                columns,
                BUILD_ID_CANDIDATES,
            )
        )

        test_result_column = (
            first_matching_column(
                columns,
                TEST_RESULT_CANDIDATES,
            )
        )

        duration_column = (
            first_matching_column(
                columns,
                DURATION_CANDIDATES,
            )
        )

        result[
            "exe_test_id_column"
        ] = test_id_column or ""

        result[
            "exe_build_id_column"
        ] = build_id_column or ""

        result[
            "exe_test_result_column"
        ] = test_result_column or ""

        result[
            "exe_duration_column"
        ] = duration_column or ""

        selected_columns = [
            column
            for column in [
                test_id_column,
                build_id_column,
                test_result_column,
                duration_column,
            ]
            if column is not None
        ]

        if not selected_columns:
            return result

        selected_columns = list(
            dict.fromkeys(
                selected_columns
            )
        )

        frame = pd.read_csv(
            path,
            usecols=selected_columns,
            low_memory=False,
        )

        if test_id_column is not None:
            result[
                "unique_tests"
            ] = int(
                frame[
                    test_id_column
                ].nunique(
                    dropna=True
                )
            )

        if build_id_column is not None:
            result[
                "unique_builds_in_exe"
            ] = int(
                frame[
                    build_id_column
                ].nunique(
                    dropna=True
                )
            )

        if test_result_column is not None:
            test_result = pd.to_numeric(
                frame[
                    test_result_column
                ],
                errors="coerce",
            )

            pass_mask = (
                test_result == 0
            )

            fail_mask = (
                test_result.isin(
                    [1, 2, 3]
                )
            )

            known_mask = (
                pass_mask
                | fail_mask
            )

            pass_rows = int(
                pass_mask.sum()
            )

            fail_rows = int(
                fail_mask.sum()
            )

            other_rows = int(
                (
                    ~known_mask
                ).sum()
            )

            denominator = (
                pass_rows
                + fail_rows
            )

            result[
                "pass_rows"
            ] = pass_rows

            result[
                "fail_rows"
            ] = fail_rows

            result[
                "other_result_rows"
            ] = other_rows

            if denominator > 0:
                result[
                    "failure_rate"
                ] = round(
                    fail_rows
                    / denominator,
                    6,
                )

        if duration_column is not None:
            duration = pd.to_numeric(
                frame[
                    duration_column
                ],
                errors="coerce",
            )

            non_null = int(
                duration.notna().sum()
            )

            missing = int(
                duration.isna().sum()
            )

            result[
                "duration_non_null"
            ] = non_null

            result[
                "duration_missing"
            ] = missing

            if non_null > 0:
                result[
                    "duration_min"
                ] = float(
                    duration.min()
                )

                result[
                    "duration_median"
                ] = float(
                    duration.median()
                )

                result[
                    "duration_max"
                ] = float(
                    duration.max()
                )

    except Exception as error:
        result[
            "exe_inspection_error"
        ] = repr(error)

    return result


# ============================================================
# BUILDS.CSV INSPECTION
# ============================================================

def inspect_builds_file(
    path: Path,
) -> Dict[str, object]:
    """Inspect timestamp information when available."""

    result: Dict[str, object] = {
        "build_timestamp_column": "",
        "build_time_non_null": None,
        "build_time_parseable": None,
        "build_time_parse_rate": None,
        "build_time_min": "",
        "build_time_max": "",
    }

    try:
        columns = read_csv_header(
            path
        )

        timestamp_column = (
            first_matching_column(
                columns,
                BUILD_TIMESTAMP_CANDIDATES,
            )
        )

        result[
            "build_timestamp_column"
        ] = timestamp_column or ""

        if timestamp_column is None:
            return result

        frame = pd.read_csv(
            path,
            usecols=[
                timestamp_column
            ],
            low_memory=False,
        )

        raw_timestamp = frame[
            timestamp_column
        ]

        non_null_count = int(
            raw_timestamp.notna().sum()
        )

        parsed = pd.to_datetime(
            raw_timestamp,
            errors="coerce",
            utc=True,
        )

        parseable_count = int(
            parsed.notna().sum()
        )

        result[
            "build_time_non_null"
        ] = non_null_count

        result[
            "build_time_parseable"
        ] = parseable_count

        if non_null_count > 0:
            result[
                "build_time_parse_rate"
            ] = round(
                parseable_count
                / non_null_count,
                6,
            )

        if parseable_count > 0:
            result[
                "build_time_min"
            ] = parsed.min().isoformat()

            result[
                "build_time_max"
            ] = parsed.max().isoformat()

    except Exception as error:
        result[
            "builds_inspection_error"
        ] = repr(error)

    return result


# ============================================================
# ANALYSIS DIRECTORY INSPECTION
# ============================================================

def inspect_analysis_dir(
    path: Path,
) -> Dict[str, object]:
    """Count high-level analysis artifacts."""

    result: Dict[str, object] = {
        "analysis_exists": False,
        "analysis_build_dirs": 0,
        "analysis_metadata_files": 0,
        "analysis_dep_files": 0,
        "analysis_tar_files": 0,
    }

    if not path.exists():
        return result

    result[
        "analysis_exists"
    ] = True

    result[
        "analysis_build_dirs"
    ] = sum(
        1
        for item in path.iterdir()
        if item.is_dir()
    )

    result[
        "analysis_metadata_files"
    ] = sum(
        1
        for _ in path.rglob(
            "metadata.csv"
        )
    )

    result[
        "analysis_dep_files"
    ] = sum(
        1
        for _ in path.rglob(
            "dep.csv"
        )
    )

    result[
        "analysis_tar_files"
    ] = sum(
        1
        for _ in path.rglob(
            "tar.csv"
        )
    )

    return result


# ============================================================
# PROJECT INVENTORY
# ============================================================

def inspect_project(
    project_dir: Path,
    raw_root: Path,
) -> tuple[
    Dict[str, object],
    List[Dict[str, object]],
]:
    """Inspect one discovered artifact/project directory."""

    key = project_key(
        project_dir,
        raw_root,
    )

    project_row: Dict[str, object] = {
        "project_key": key,
        "project_dir": str(
            project_dir.relative_to(
                REPO_ROOT
            )
        ),
    }

    file_rows: List[
        Dict[str, object]
    ] = []

    for filename in EXPECTED_FILES:
        file_path = (
            project_dir
            / filename
        )

        column_name = (
            "has_"
            + filename.replace(
                ".csv",
                "",
            )
        )

        exists = file_path.exists()

        project_row[
            column_name
        ] = exists

        if not exists:
            continue

        file_info = (
            inspect_expected_file(
                key,
                file_path,
            )
        )

        file_rows.append(
            file_info
        )

        row_column = (
            filename.replace(
                ".csv",
                "",
            )
            + "_rows"
        )

        project_row[
            row_column
        ] = file_info[
            "row_count"
        ]

    # --------------------------------------------------------
    # Semantic inspection when known columns happen to exist.
    # --------------------------------------------------------

    exe_path = (
        project_dir
        / "exe.csv"
    )

    if exe_path.exists():
        project_row.update(
            inspect_exe_file(
                exe_path
            )
        )

    builds_path = (
        project_dir
        / "builds.csv"
    )

    if builds_path.exists():
        project_row.update(
            inspect_builds_file(
                builds_path
            )
        )

    # --------------------------------------------------------
    # analysis/
    # --------------------------------------------------------

    analysis_path = (
        project_dir
        / "analysis"
    )

    project_row.update(
        inspect_analysis_dir(
            analysis_path
        )
    )

    # --------------------------------------------------------
    # Structural screening flag only.
    #
    # This is NOT final pilot eligibility.
    # --------------------------------------------------------

    project_row[
        "core_triplet_present"
    ] = all(
        (
            project_dir
            / filename
        ).exists()
        for filename in [
            "dataset.csv",
            "exe.csv",
            "builds.csv",
        ]
    )

    return (
        project_row,
        file_rows,
    )


# ============================================================
# SCHEMA VARIANTS
# ============================================================

def build_schema_variants(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize schema differences between projects."""

    if file_inventory.empty:
        return pd.DataFrame()

    valid = file_inventory[
        file_inventory[
            "read_status"
        ] == "OK"
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    grouped_rows = []

    grouped = valid.groupby(
        [
            "artifact",
            "schema_signature",
            "columns",
        ],
        dropna=False,
    )

    for (
        artifact,
        signature,
        columns,
    ), group in grouped:
        projects = sorted(
            group[
                "project_key"
            ].tolist()
        )

        grouped_rows.append(
            {
                "artifact": artifact,
                "schema_signature": (
                    signature
                ),
                "project_count": len(
                    projects
                ),
                "projects": json.dumps(
                    projects,
                    ensure_ascii=False,
                ),
                "columns": columns,
            }
        )

    return pd.DataFrame(
        grouped_rows
    ).sort_values(
        [
            "artifact",
            "project_count",
        ],
        ascending=[
            True,
            False,
        ],
    )


# ============================================================
# OUTPUT
# ============================================================

def write_outputs(
    project_rows: List[
        Dict[str, object]
    ],
    file_rows: List[
        Dict[str, object]
    ],
    output_root: Path,
    raw_root: Path,
) -> None:
    """Write inventory outputs."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    projects = pd.DataFrame(
        project_rows
    )

    files = pd.DataFrame(
        file_rows
    )

    if not projects.empty:
        projects = projects.sort_values(
            "project_key"
        )

    if not files.empty:
        files = files.sort_values(
            [
                "project_key",
                "artifact",
            ]
        )

    projects_path = (
        output_root
        / "project_inventory.csv"
    )

    files_path = (
        output_root
        / "file_inventory.csv"
    )

    schemas_path = (
        output_root
        / "schema_variants.csv"
    )

    summary_path = (
        output_root
        / "inventory_summary.json"
    )

    projects.to_csv(
        projects_path,
        index=False,
    )

    files.to_csv(
        files_path,
        index=False,
    )

    schemas = (
        build_schema_variants(
            files
        )
    )

    schemas.to_csv(
        schemas_path,
        index=False,
    )

    core_ready = 0

    if (
        not projects.empty
        and "core_triplet_present"
        in projects.columns
    ):
        core_ready = int(
            projects[
                "core_triplet_present"
            ].fillna(False).sum()
        )

    artifact_counts = {}

    if not files.empty:
        artifact_counts = {
            str(key): int(value)
            for key, value
            in files[
                "artifact"
            ].value_counts().items()
        }

    summary = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "raw_root": str(
            raw_root
        ),
        "project_groups_discovered": (
            len(projects)
        ),
        "core_triplet_present_count": (
            core_ready
        ),
        "artifact_counts": (
            artifact_counts
        ),
        "outputs": {
            "project_inventory": str(
                projects_path
            ),
            "file_inventory": str(
                files_path
            ),
            "schema_variants": str(
                schemas_path
            ),
        },
        "notes": [
            (
                "Project discovery is based on "
                "directories containing dataset.csv, "
                "exe.csv, or builds.csv."
            ),
            (
                "Detected column mappings are "
                "inventory hints only and do not "
                "freeze the thesis canonical schema."
            ),
            (
                "core_triplet_present is a "
                "structural flag, not final "
                "pilot eligibility."
            ),
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 64)
    print("TCP-CI INVENTORY COMPLETE")
    print("=" * 64)

    print(
        "Project/artifact groups:",
        len(projects),
    )

    print(
        "Groups with dataset.csv + "
        "exe.csv + builds.csv:",
        core_ready,
    )

    print()
    print("Artifact counts:")

    for artifact, count in sorted(
        artifact_counts.items()
    ):
        print(
            f"  {artifact}: {count}"
        )

    print()
    print("Outputs:")

    print(
        f"  {projects_path}"
    )

    print(
        f"  {files_path}"
    )

    print(
        f"  {schemas_path}"
    )

    print(
        f"  {summary_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    raw_root = (
        args.raw_root.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    if not raw_root.exists():
        raise FileNotFoundError(
            "TCP-CI raw root does not exist: "
            f"{raw_root}"
        )

    if not raw_root.is_dir():
        raise NotADirectoryError(
            "TCP-CI raw root is not a directory: "
            f"{raw_root}"
        )

    print(
        "TCP-CI raw root:"
    )

    print(
        f"  {raw_root}"
    )

    print()
    print(
        "Discovering artifact/project directories..."
    )

    project_dirs = (
        discover_project_dirs(
            raw_root
        )
    )

    if not project_dirs:
        raise RuntimeError(
            "No TCP-CI project/artifact directories "
            "were discovered. Expected at least one "
            "directory containing dataset.csv, "
            "exe.csv, or builds.csv."
        )

    print(
        "Discovered:",
        len(project_dirs),
    )

    project_rows = []
    file_rows = []

    for index, project_dir in enumerate(
        project_dirs,
        start=1,
    ):
        key = project_key(
            project_dir,
            raw_root,
        )

        print(
            f"[{index}/{len(project_dirs)}] "
            f"{key}"
        )

        (
            project_row,
            project_file_rows,
        ) = inspect_project(
            project_dir,
            raw_root,
        )

        project_rows.append(
            project_row
        )

        file_rows.extend(
            project_file_rows
        )

    write_outputs(
        project_rows=project_rows,
        file_rows=file_rows,
        output_root=output_root,
        raw_root=raw_root,
    )


if __name__ == "__main__":
    main()