from pathlib import Path

import pandas as pd
import yaml

from src.tcp_ci.build_history_features import (
    FEATURE_COLUMNS,
    SUPPORT_COLUMNS,
    compute_history_features,
)


KEY_COLUMNS = [
    "project_id",
    "build_id",
    "test_id",
]


CHECK_COLUMNS = (
    SUPPORT_COLUMNS
    + FEATURE_COLUMNS
)


def make_project_data(
    project_id="p1",
):
    return pd.DataFrame(
        {
            "project_id": [
                project_id,
                project_id,
                project_id,
                project_id,
                project_id,
                project_id,
                project_id,
                project_id,
            ],
            "build_id": [
                "1",
                "1",
                "2",
                "2",
                "3",
                "3",
                "4",
                "4",
            ],
            "build_started_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                ]
            ),
            "build_index": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            "test_id": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "execution_time_ms": [
                100,
                1000,
                200,
                1100,
                300,
                1200,
                400,
                1300,
            ],
            "failed": [
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
            ],
        }
    )


def history_for_row(
    dataframe,
    project_id,
    build_id,
    test_id,
):
    row = dataframe[
        (dataframe["project_id"] == project_id)
        & (dataframe["build_id"] == build_id)
        & (dataframe["test_id"] == test_id)
    ]

    assert len(row) == 1

    return (
        row.iloc[0][CHECK_COLUMNS]
        .copy()
    )


def sorted_feature_view(
    dataframe,
):
    columns = (
        KEY_COLUMNS
        + CHECK_COLUMNS
    )

    return (
        dataframe[columns]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )


def test_current_row_outcome_and_cost_do_not_change_history_features():
    base = make_project_data()

    original = compute_history_features(
        base
    )

    mutated = base.copy()

    mask = (
        (mutated["project_id"] == "p1")
        & (mutated["build_id"] == "3")
        & (mutated["test_id"] == "A")
    )

    assert mask.sum() == 1

    # Deliberately alter post-execution information
    # from the target row itself.
    mutated.loc[
        mask,
        "failed",
    ] = 1

    mutated.loc[
        mask,
        "execution_time_ms",
    ] = 999999

    changed = compute_history_features(
        mutated
    )

    expected = history_for_row(
        original,
        "p1",
        "3",
        "A",
    )

    actual = history_for_row(
        changed,
        "p1",
        "3",
        "A",
    )

    pd.testing.assert_series_equal(
        expected,
        actual,
        check_names=False,
    )


def test_future_outcome_and_cost_do_not_change_past_features():
    base = make_project_data()

    original = compute_history_features(
        base
    )

    mutated = base.copy()

    future = (
        mutated["build_index"] == 4
    )

    # Change every post-execution value in the future build.
    mutated.loc[
        future,
        "failed",
    ] = (
        1
        - mutated.loc[
            future,
            "failed",
        ]
    )

    mutated.loc[
        future,
        "execution_time_ms",
    ] = 999999

    changed = compute_history_features(
        mutated
    )

    expected = sorted_feature_view(
        original[
            original[
                "build_index"
            ] <= 3
        ]
    )

    actual = sorted_feature_view(
        changed[
            changed[
                "build_index"
            ] <= 3
        ]
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_prefix_features_equal_full_dataset_features():
    base = make_project_data()

    full = compute_history_features(
        base
    )

    prefix_input = base[
        base["build_index"] <= 3
    ].copy()

    prefix = compute_history_features(
        prefix_input
    )

    expected = sorted_feature_view(
        full[
            full[
                "build_index"
            ] <= 3
        ]
    )

    actual = sorted_feature_view(
        prefix
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_history_is_isolated_between_projects():
    project_1 = make_project_data(
        "p1"
    )

    project_2 = make_project_data(
        "p2"
    )

    # Give project 2 a very different history while
    # deliberately reusing the same test IDs A and B.
    project_2["failed"] = (
        1 - project_2["failed"]
    )

    project_2[
        "execution_time_ms"
    ] = (
        project_2[
            "execution_time_ms"
        ]
        + 50000
    )

    alone = compute_history_features(
        project_1
    )

    combined = compute_history_features(
        pd.concat(
            [
                project_1,
                project_2,
            ],
            ignore_index=True,
        )
    )

    combined_p1 = combined[
        combined["project_id"] == "p1"
    ]

    expected = sorted_feature_view(
        alone
    )

    actual = sorted_feature_view(
        combined_p1
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_input_row_order_does_not_change_history_features():
    base = make_project_data()

    original = compute_history_features(
        base
    )

    shuffled = (
        base.sample(
            frac=1.0,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    reordered = compute_history_features(
        shuffled
    )

    expected = sorted_feature_view(
        original
    )

    actual = sorted_feature_view(
        reordered
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_feature_config_excludes_current_post_execution_fields():
    path = Path(
        "configs/history_features_v1.yaml"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    configured = set(
        config[
            "feature_columns"
        ]
    )

    forbidden = {
        "failed",
        "execution_time_ms",
        "verdict_raw",
        "duration_raw",
    }

    assert configured.isdisjoint(
        forbidden
    )

    assert (
        config[
            "temporal_rule"
        ][
            "same_build_outcome_allowed"
        ]
        is False
    )

    assert (
        config[
            "temporal_rule"
        ][
            "future_build_outcome_allowed"
        ]
        is False
    )


def test_schema_marks_history_features_pre_execution_and_safe():
    schema_path = Path(
        "configs/tcp_ci_schema_v1.yaml"
    )

    history_config_path = Path(
        "configs/history_features_v1.yaml"
    )

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        schema = yaml.safe_load(file)

    with history_config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        history_config = yaml.safe_load(
            file
        )

    schema_history = (
        schema[
            "candidate_features"
        ][
            "history"
        ]
    )

    for feature in history_config[
        "feature_columns"
    ]:
        assert feature in schema_history

        definition = (
            schema_history[feature]
        )

        assert (
            definition[
                "available_at"
            ]
            == "pre_test_execution"
        )

        assert (
            definition[
                "leakage_risk"
            ]
            == "SAFE"
        )