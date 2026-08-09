"""Tests for the legacy synthetic-data generator.

These tests verify that the repository can regenerate the
synthetic dataset used by the original graduation project
without modifying the frozen historical artifact.
"""

from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal


# ============================================================
# REPOSITORY PATH
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure the repository root can be imported when pytest
# is executed from different working directories.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from legacy.generate_synthetic_data import (  # noqa: E402
    OUTPUT_PATH,
    generate_synthetic_data,
)


FROZEN_DATASET_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "synthetic"
    / "synthetic_data.csv"
)


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


# ============================================================
# BASIC DATASET STRUCTURE
# ============================================================

def test_synthetic_dataset_shape():
    """The regenerated dataset must contain 5000 rows and 9 columns."""

    regenerated = generate_synthetic_data()

    assert regenerated.shape == (
        5000,
        9,
    )


def test_synthetic_dataset_columns():
    """The regenerated dataset must preserve the legacy schema."""

    regenerated = generate_synthetic_data()

    assert list(
        regenerated.columns
    ) == EXPECTED_COLUMNS


def test_synthetic_dataset_has_expected_sprints():
    """The dataset must contain 50 Sprints."""

    regenerated = generate_synthetic_data()

    assert regenerated[
        "Sprint_ID"
    ].nunique() == 50

    assert regenerated[
        "Sprint_ID"
    ].min() == 1

    assert regenerated[
        "Sprint_ID"
    ].max() == 50


def test_each_sprint_contains_100_test_cases():
    """Every Sprint must contain exactly 100 test cases."""

    regenerated = generate_synthetic_data()

    rows_per_sprint = (
        regenerated
        .groupby("Sprint_ID")
        .size()
    )

    assert (
        rows_per_sprint == 100
    ).all()


def test_sprint_testcase_pairs_are_unique():
    """Each Sprint/TestCase pair must occur exactly once."""

    regenerated = generate_synthetic_data()

    duplicated_pairs = regenerated.duplicated(
        subset=[
            "Sprint_ID",
            "TestCase_ID",
        ]
    )

    assert not duplicated_pairs.any()


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def test_synthetic_label_distribution():
    """The regenerated dataset must reproduce the final labels."""

    regenerated = generate_synthetic_data()

    label_counts = (
        regenerated["Label"]
        .value_counts()
        .to_dict()
    )

    assert label_counts == {
        0: 4386,
        1: 614,
    }


def test_synthetic_positive_rate():
    """The positive-label rate must be 12.28%."""

    regenerated = generate_synthetic_data()

    positive_rate = (
        regenerated["Label"]
        .mean()
    )

    assert positive_rate == 614 / 5000


# ============================================================
# FROZEN ARTIFACT COMPARISON
# ============================================================

def test_regenerated_dataset_matches_frozen_dataset():
    """Regenerated data must exactly match the frozen CSV.

    Data types are ignored because CSV loading may infer a
    slightly different dtype, but column order, row order,
    shape, and all values must match exactly.
    """

    assert FROZEN_DATASET_PATH.exists(), (
        "Frozen dataset was not found at: "
        f"{FROZEN_DATASET_PATH}"
    )

    frozen = pd.read_csv(
        FROZEN_DATASET_PATH
    )

    regenerated = (
        generate_synthetic_data()
        .reset_index(drop=True)
    )

    frozen = frozen.reset_index(
        drop=True
    )

    assert_frame_equal(
        frozen,
        regenerated,
        check_dtype=False,
        check_exact=True,
    )


# ============================================================
# FROZEN ARTIFACT PROTECTION
# ============================================================

def test_generator_output_does_not_overwrite_frozen_dataset():
    """Generated output must be written outside data/raw."""

    expected_output = (
        REPO_ROOT
        / "data"
        / "interim"
        / "synthetic_regenerated.csv"
    )

    assert OUTPUT_PATH.resolve() == (
        expected_output.resolve()
    )

    assert OUTPUT_PATH.resolve() != (
        FROZEN_DATASET_PATH.resolve()
    )