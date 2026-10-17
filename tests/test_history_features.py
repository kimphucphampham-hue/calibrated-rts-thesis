import pandas as pd

from src.tcp_ci.build_history_features import (
    compute_changed_entity_count,
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



def make_window_input():
    rows = []
    failures = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1]

    for build_index in range(1, 12):
        rows.append(
            {
                "project_id": "p",
                "build_id": str(build_index),
                "build_started_at": pd.Timestamp(
                    f"2026-02-{build_index:02d}T00:00:00Z"
                ),
                "build_index": build_index,
                "test_id": "A",
                "execution_time_ms": (
                    999999
                    if build_index == 11
                    else build_index * 100
                ),
                "failed": failures[build_index - 1],
            }
        )

    return pd.DataFrame(rows)


def test_recent_failure_density_uses_previous_10_only():
    result = compute_history_features(make_window_input())

    before_window = result[
        result["build_index"] == 10
    ].iloc[0]

    target = result[
        result["build_index"] == 11
    ].iloc[0]

    assert pd.isna(
        before_window["recent_failure_density_10"]
    )
    assert target["recent_failure_density_10"] == 0.4


def test_execution_time_trend_uses_previous_10_only():
    result = compute_history_features(make_window_input())

    before_window = result[
        result["build_index"] == 10
    ].iloc[0]

    target = result[
        result["build_index"] == 11
    ].iloc[0]

    assert pd.isna(
        before_window["execution_time_trend_10"]
    )
    assert target["execution_time_trend_10"] == 100.0


def test_changed_entity_count_follows_frozen_policy(tmp_path):
    canonical = pd.DataFrame(
        {
            "build_id": ["b1", "b2", "b3", "b4"],
            "build_started_at": pd.to_datetime(
                [
                    "2026-03-10T00:00:00Z",
                    "2026-03-10T00:00:00Z",
                    "2026-03-10T00:00:00Z",
                    "2026-03-10T00:00:00Z",
                ]
            ),
        }
    )

    builds = pd.DataFrame(
        {
            "id": ["b1", "b2", "b3", "b4"],
            "commits": [
                "c1#c2",
                "c3",
                "missing-commit",
                pd.NA,
            ],
            "started_at": [
                "2026-03-10T00:00:00Z",
                "2026-03-10T00:00:00Z",
                "2026-03-10T00:00:00Z",
                "2026-03-10T00:00:00Z",
            ],
        }
    )

    changes = pd.DataFrame(
        {
            "Commit": ["c1", "c1", "c2", "c2", "c3"],
            "CommitDate": [
                "2026-03-09T00:00:00Z",
                "2026-03-09T00:00:00Z",
                "2026-03-09T00:00:00Z",
                "2026-03-09T00:00:00Z",
                "2026-03-11T00:00:00Z",
            ],
            "EntityId": [1, 1, 2, 3, 4],
        }
    )

    builds.to_csv(tmp_path / "builds.csv", index=False)
    changes.to_csv(
        tmp_path / "entity_change_history.csv",
        index=False,
    )

    result = compute_changed_entity_count(
        canonical=canonical,
        raw_project_dir=tmp_path,
    )

    values = result.set_index(
        "build_id"
    )["changed_entity_count"]

    assert values["b1"] == 3.0
    assert pd.isna(values["b2"])
    assert pd.isna(values["b3"])
    assert pd.isna(values["b4"])
