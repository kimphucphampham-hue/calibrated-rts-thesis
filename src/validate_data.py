"""Validate and profile the legacy synthetic dataset.

Run from the repository root:

    python src/validate_data.py

Outputs:
    results/legacy/synthetic_data_profile.csv
    results/legacy/synthetic_validation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# File hiện tại nằm ở:
# calibrated-rts-thesis/src/validate_data.py
#
# parents[1] sẽ lấy thư mục:
# calibrated-rts-thesis/
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "synthetic"
    / "synthetic_data.csv"
)

OUTPUT_DIR = REPO_ROOT / "results" / "legacy"

PROFILE_OUTPUT_PATH = (
    OUTPUT_DIR / "synthetic_data_profile.csv"
)

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIR / "synthetic_validation.json"
)


# ============================================================
# EXPECTED DATASET SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "Sprint_ID",
    "TestCase_ID",
    "Code_Churn",
    "Cyclomatic_Complexity",
    "Dependencies",
    "Failure_History",
    "Execution_Time_ms",
    "Test_Age",
    "Label",
]

IDENTIFIER_COLUMNS = [
    "Sprint_ID",
    "TestCase_ID",
]

FEATURE_COLUMNS = [
    "Code_Churn",
    "Cyclomatic_Complexity",
    "Dependencies",
    "Failure_History",
    "Execution_Time_ms",
    "Test_Age",
]

LABEL_COLUMN = "Label"

NUMERIC_COLUMNS = (
    IDENTIFIER_COLUMNS
    + FEATURE_COLUMNS
    + [LABEL_COLUMN]
)


# ============================================================
# EXPECTED CHARACTERISTICS OF THE FROZEN ARTIFACT
# ============================================================

EXPECTED_ROW_COUNT = 5_000
EXPECTED_SPRINT_COUNT = 50
EXPECTED_TEST_COUNT = 100

# Báo cáo mô tả phân phối khoảng 10% nhãn 1.
DOCUMENTED_POSITIVE_RATE = 0.10

# Nếu tỷ lệ thực tế lệch quá 0,5 điểm phần trăm,
# script sẽ ghi cảnh báo.
POSITIVE_RATE_TOLERANCE = 0.005


def to_python_value(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-compatible Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def create_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create one profiling row for every dataset column."""

    profile_rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]

        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()

            if not non_null_series.empty:
                row.update(
                    {
                        "minimum": to_python_value(
                            non_null_series.min()
                        ),
                        "maximum": to_python_value(
                            non_null_series.max()
                        ),
                        "mean": to_python_value(
                            non_null_series.mean()
                        ),
                        "median": to_python_value(
                            non_null_series.median()
                        ),
                        "standard_deviation": to_python_value(
                            non_null_series.std()
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "minimum": None,
                        "maximum": None,
                        "mean": None,
                        "median": None,
                        "standard_deviation": None,
                    }
                )
        else:
            row.update(
                {
                    "minimum": None,
                    "maximum": None,
                    "mean": None,
                    "median": None,
                    "standard_deviation": None,
                }
            )

        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, quality, identifiers, labels and value ranges."""

    errors: list[str] = []
    warnings: list[str] = []
    passed_checks: list[str] = []

    actual_columns = df.columns.tolist()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    # --------------------------------------------------------
    # Check 1: Required columns
    # --------------------------------------------------------

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
    else:
        passed_checks.append(
            "All required columns are present."
        )

    if unexpected_columns:
        warnings.append(
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )
    else:
        passed_checks.append(
            "No unexpected columns were found."
        )

    # --------------------------------------------------------
    # Check 2: Column order
    # --------------------------------------------------------

    if not missing_columns:
        if actual_columns == EXPECTED_COLUMNS:
            passed_checks.append(
                "Column order matches the expected schema."
            )
        else:
            warnings.append(
                "All required columns exist, but their order "
                "differs from the expected schema."
            )

    # --------------------------------------------------------
    # Check 3: Missing values
    # --------------------------------------------------------

    missing_value_counts = (
        df.isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    total_missing_values = int(df.isna().sum().sum())

    if total_missing_values > 0:
        errors.append(
            f"Dataset contains {total_missing_values} "
            "missing values."
        )
    else:
        passed_checks.append(
            "No missing values were found."
        )

    # --------------------------------------------------------
    # Check 4: Duplicate rows
    # --------------------------------------------------------

    duplicate_row_count = int(df.duplicated().sum())

    if duplicate_row_count > 0:
        errors.append(
            f"Dataset contains {duplicate_row_count} "
            "fully duplicated rows."
        )
    else:
        passed_checks.append(
            "No fully duplicated rows were found."
        )

    # --------------------------------------------------------
    # Check 5: Numeric columns
    # --------------------------------------------------------

    available_numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    non_numeric_columns: list[str] = []

    for column in available_numeric_columns:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_numeric_count = int(
            converted.isna().sum()
            - df[column].isna().sum()
        )

        if invalid_numeric_count > 0:
            non_numeric_columns.append(column)
            errors.append(
                f"Column '{column}' contains "
                f"{invalid_numeric_count} non-numeric values."
            )

    if not non_numeric_columns:
        passed_checks.append(
            "All expected numeric columns contain numeric values."
        )

    # --------------------------------------------------------
    # Check 6: Infinite values
    # --------------------------------------------------------

    infinite_value_count = 0

    if available_numeric_columns:
        numeric_frame = df[
            available_numeric_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        infinite_value_count = int(
            np.isinf(numeric_frame.to_numpy()).sum()
        )

    if infinite_value_count > 0:
        errors.append(
            f"Dataset contains {infinite_value_count} "
            "infinite numeric values."
        )
    else:
        passed_checks.append(
            "No infinite numeric values were found."
        )

    # --------------------------------------------------------
    # Check 7: Label values
    # --------------------------------------------------------

    label_distribution: dict[str, int] = {}
    positive_count = 0
    negative_count = 0
    positive_rate: float | None = None

    if LABEL_COLUMN in df.columns:
        label_values = sorted(
            df[LABEL_COLUMN]
            .dropna()
            .unique()
            .tolist()
        )

        invalid_label_values = [
            to_python_value(value)
            for value in label_values
            if value not in (0, 1)
        ]

        if invalid_label_values:
            errors.append(
                "Label contains values other than 0 and 1: "
                + str(invalid_label_values)
            )
        else:
            passed_checks.append(
                "Label contains only binary values 0 and 1."
            )

        value_counts = (
            df[LABEL_COLUMN]
            .value_counts(dropna=False)
            .sort_index()
        )

        label_distribution = {
            str(to_python_value(label)): int(count)
            for label, count in value_counts.items()
        }

        positive_count = int(
            (df[LABEL_COLUMN] == 1).sum()
        )

        negative_count = int(
            (df[LABEL_COLUMN] == 0).sum()
        )

        if len(df) > 0:
            positive_rate = positive_count / len(df)

            difference = abs(
                positive_rate
                - DOCUMENTED_POSITIVE_RATE
            )

            if difference > POSITIVE_RATE_TOLERANCE:
                warnings.append(
                    "Actual positive-label rate is "
                    f"{positive_rate:.2%}, while the report "
                    f"describes approximately "
                    f"{DOCUMENTED_POSITIVE_RATE:.2%}."
                )
            else:
                passed_checks.append(
                    "Positive-label rate is close to the "
                    "documented 10% distribution."
                )

    # --------------------------------------------------------
    # Check 8: Identifiers
    # --------------------------------------------------------

    sprint_count: int | None = None
    test_count: int | None = None

    if "Sprint_ID" in df.columns:
        sprint_count = int(
            df["Sprint_ID"].nunique(dropna=True)
        )

        non_positive_sprint_count = int(
            (df["Sprint_ID"] <= 0).sum()
        )

        if non_positive_sprint_count > 0:
            errors.append(
                f"Sprint_ID contains "
                f"{non_positive_sprint_count} "
                "values less than or equal to zero."
            )
        else:
            passed_checks.append(
                "All Sprint_ID values are positive."
            )

    if "TestCase_ID" in df.columns:
        test_count = int(
            df["TestCase_ID"].nunique(dropna=True)
        )

        non_positive_test_count = int(
            (df["TestCase_ID"] <= 0).sum()
        )

        if non_positive_test_count > 0:
            errors.append(
                f"TestCase_ID contains "
                f"{non_positive_test_count} "
                "values less than or equal to zero."
            )
        else:
            passed_checks.append(
                "All TestCase_ID values are positive."
            )

    # Kiểm tra cùng một test có bị lặp trong cùng một Sprint không.
    duplicate_test_in_sprint_count = 0

    if {
        "Sprint_ID",
        "TestCase_ID",
    }.issubset(df.columns):
        duplicate_test_in_sprint_count = int(
            df.duplicated(
                subset=[
                    "Sprint_ID",
                    "TestCase_ID",
                ]
            ).sum()
        )

        if duplicate_test_in_sprint_count > 0:
            errors.append(
                "There are "
                f"{duplicate_test_in_sprint_count} duplicated "
                "Sprint_ID/TestCase_ID combinations."
            )
        else:
            passed_checks.append(
                "Each TestCase_ID appears at most once "
                "within each Sprint_ID."
            )

    # --------------------------------------------------------
    # Check 9: Duration
    # --------------------------------------------------------

    invalid_duration_count = 0

    if "Execution_Time_ms" in df.columns:
        invalid_duration_count = int(
            (df["Execution_Time_ms"] <= 0).sum()
        )

        if invalid_duration_count > 0:
            errors.append(
                "Execution_Time_ms contains "
                f"{invalid_duration_count} values "
                "less than or equal to zero."
            )
        else:
            passed_checks.append(
                "All execution times are greater than zero."
            )

    # --------------------------------------------------------
    # Check 10: Other non-negative feature values
    # --------------------------------------------------------

    non_negative_columns = [
        "Code_Churn",
        "Dependencies",
        "Failure_History",
        "Test_Age",
    ]

    for column in non_negative_columns:
        if column not in df.columns:
            continue

        negative_count = int(
            (df[column] < 0).sum()
        )

        if negative_count > 0:
            errors.append(
                f"Column '{column}' contains "
                f"{negative_count} negative values."
            )
        else:
            passed_checks.append(
                f"Column '{column}' has no negative values."
            )

    if "Cyclomatic_Complexity" in df.columns:
        invalid_complexity_count = int(
            (df["Cyclomatic_Complexity"] <= 0).sum()
        )

        if invalid_complexity_count > 0:
            warnings.append(
                "Cyclomatic_Complexity contains "
                f"{invalid_complexity_count} values "
                "less than or equal to zero."
            )
        else:
            passed_checks.append(
                "All Cyclomatic_Complexity values "
                "are greater than zero."
            )

    # --------------------------------------------------------
    # Check 11: Frozen artifact expected size
    # --------------------------------------------------------

    row_count = int(len(df))
    column_count = int(len(df.columns))

    if row_count != EXPECTED_ROW_COUNT:
        warnings.append(
            f"Expected {EXPECTED_ROW_COUNT} rows, "
            f"but found {row_count}."
        )
    else:
        passed_checks.append(
            f"Row count matches the expected "
            f"{EXPECTED_ROW_COUNT} rows."
        )

    if sprint_count is not None:
        if sprint_count != EXPECTED_SPRINT_COUNT:
            warnings.append(
                f"Expected {EXPECTED_SPRINT_COUNT} unique "
                f"Sprints, but found {sprint_count}."
            )
        else:
            passed_checks.append(
                f"Sprint count matches the expected "
                f"{EXPECTED_SPRINT_COUNT} Sprints."
            )

    if test_count is not None:
        if test_count != EXPECTED_TEST_COUNT:
            warnings.append(
                f"Expected {EXPECTED_TEST_COUNT} unique "
                f"test cases, but found {test_count}."
            )
        else:
            passed_checks.append(
                f"Test count matches the expected "
                f"{EXPECTED_TEST_COUNT} test cases."
            )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    result: dict[str, Any] = {
        "status": status,
        "dataset_path": str(DATA_PATH),
        "row_count": row_count,
        "column_count": column_count,
        "actual_columns": actual_columns,
        "expected_columns": EXPECTED_COLUMNS,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "sprint_count": sprint_count,
        "unique_test_count": test_count,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_test_in_sprint_count": (
            duplicate_test_in_sprint_count
        ),
        "total_missing_values": total_missing_values,
        "missing_values_by_column": missing_value_counts,
        "infinite_value_count": infinite_value_count,
        "invalid_duration_count": invalid_duration_count,
        "label_distribution": label_distribution,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "documented_positive_rate": (
            DOCUMENTED_POSITIVE_RATE
        ),
        "passed_checks": passed_checks,
        "warnings": warnings,
        "errors": errors,
    }

    return result


def print_summary(
    validation_result: dict[str, Any],
) -> None:
    """Print a readable validation summary to the Terminal."""

    print()
    print("=" * 65)
    print("SYNTHETIC DATASET VALIDATION SUMMARY")
    print("=" * 65)

    print(
        f"Status: {validation_result['status']}"
    )

    print(
        "Dataset: "
        f"{validation_result['dataset_path']}"
    )

    print(
        "Shape: "
        f"{validation_result['row_count']} rows x "
        f"{validation_result['column_count']} columns"
    )

    print(
        "Unique Sprints: "
        f"{validation_result['sprint_count']}"
    )

    print(
        "Unique Test Cases: "
        f"{validation_result['unique_test_count']}"
    )

    print(
        "Missing values: "
        f"{validation_result['total_missing_values']}"
    )

    print(
        "Duplicated rows: "
        f"{validation_result['duplicate_row_count']}"
    )

    print(
        "Label distribution: "
        f"{validation_result['label_distribution']}"
    )

    positive_rate = validation_result[
        "positive_rate"
    ]

    if positive_rate is not None:
        print(
            f"Positive-label rate: "
            f"{positive_rate:.2%}"
        )

    if validation_result["warnings"]:
        print()
        print("WARNINGS:")

        for warning in validation_result["warnings"]:
            print(f"  - {warning}")

    if validation_result["errors"]:
        print()
        print("ERRORS:")

        for error in validation_result["errors"]:
            print(f"  - {error}")

    print()
    print(
        "Profile CSV: "
        f"{PROFILE_OUTPUT_PATH}"
    )

    print(
        "Validation JSON: "
        f"{VALIDATION_OUTPUT_PATH}"
    )

    print("=" * 65)


def main() -> int:
    """Run dataset loading, profiling and validation."""

    print(
        f"[INFO] Repository root: {REPO_ROOT}"
    )

    print(
        f"[INFO] Reading dataset: {DATA_PATH}"
    )

    if not DATA_PATH.exists():
        print(
            "[ERROR] Synthetic dataset was not found.",
            file=sys.stderr,
        )

        print(
            "[ERROR] Expected location: "
            f"{DATA_PATH}",
            file=sys.stderr,
        )

        return 1

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(
            "[ERROR] Could not read the CSV file: "
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_df = create_column_profile(df)
    validation_result = validate_dataset(df)

    profile_df.to_csv(
        PROFILE_OUTPUT_PATH,
        index=False,
    )

    with VALIDATION_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            validation_result,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(validation_result)

    if validation_result["errors"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())