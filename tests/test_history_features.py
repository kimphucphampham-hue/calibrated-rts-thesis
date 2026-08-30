import pandas as pd

from src.tcp_ci.build_history_features import (
    compute_history_features,
)


def make_input():
    return pd.DataFrame(
        {
            "project_id": [
                "p",
                "p",
                "p",
                "p",
                "p",
            ],
            "build_id": [
                "1",
                "2",
                "3",
                "4",
                "5",
            ],
            "build_started_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
            "build_index": [
                1,
                2,
                3,
                4,
                5,
            ],
            "test_id": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "execution_time_ms": [
                100,
                200,
                300,
                400,
                500,
            ],
            "failed": [
                1,
                0,
                1,
                0,
                0,
            ],
        }
    )


def test_cold_start_features():
    result = compute_history_features(
        make_input()
    )

    first_a = result[
        (
            result["test_id"] == "A"
        )
        &
        (
            result["build_index"] == 1
        )
    ].iloc[0]

    assert (
        first_a[
            "history_available"
        ]
        == 0
    )

    assert (
        first_a[
            "prior_execution_count"
        ]
        == 0
    )

    assert pd.isna(
        first_a[
            "failure_rate_history"
        ]
    )

    assert pd.isna(
        first_a[
            "recent_failure"
        ]
    )

    assert (
        first_a[
            "test_age_builds"
        ]
        == 0
    )


def test_history_uses_only_prior_rows():
    result = compute_history_features(
        make_input()
    )

    a_build_3 = result[
        (
            result["test_id"] == "A"
        )
        &
        (
            result["build_index"] == 3
        )
    ].iloc[0]

    assert (
        a_build_3[
            "prior_execution_count"
        ]
        == 1
    )

    assert (
        a_build_3[
            "prior_failure_count"
        ]
        == 1
    )

    assert (
        a_build_3[
            "failure_rate_history"
        ]
        == 1.0
    )

    assert (
        a_build_3[
            "recent_failure"
        ]
        == 1
    )

    assert (
        a_build_3[
            "execution_time_mean_history_ms"
        ]
        == 100.0
    )

    assert (
        a_build_3[
            "time_since_last_failure_builds"
        ]
        == 2
    )


def test_streaks_are_prior_only():
    result = compute_history_features(
        make_input()
    )

    a_build_5 = result[
        (
            result["test_id"] == "A"
        )
        &
        (
            result["build_index"] == 5
        )
    ].iloc[0]

    assert (
        a_build_5[
            "prior_execution_count"
        ]
        == 2
    )

    assert (
        a_build_5[
            "prior_failure_count"
        ]
        == 2
    )

    assert (
        a_build_5[
            "consecutive_failures"
        ]
        == 2
    )

    assert (
        a_build_5[
            "consecutive_passes"
        ]
        == 0
    )

    assert (
        a_build_5[
            "execution_time_mean_history_ms"
        ]
        == 200.0
    )

    assert (
        a_build_5[
            "time_since_last_failure_builds"
        ]
        == 2
    )