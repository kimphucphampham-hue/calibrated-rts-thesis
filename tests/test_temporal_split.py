import pandas as pd

from src.tcp_ci.split_by_build import (
    build_split_table,
)


def make_data():
    rows = []

    for build_index in range(
        1,
        11,
    ):
        for test_id in [
            "A",
            "B",
        ]:
            rows.append(
                {
                    "project_id": "p",
                    "build_id":
                        str(build_index),

                    "build_started_at":
                        (
                            pd.Timestamp(
                                "2026-01-01",
                                tz="UTC",
                            )
                            + pd.Timedelta(
                                days=build_index,
                            )
                        ),

                    "build_index":
                        build_index,

                    "test_id":
                        test_id,

                    "failed":
                        int(
                            (
                                build_index
                                + (
                                    test_id
                                    == "B"
                                )
                            )
                            % 3
                            == 0
                        ),

                    "execution_time_ms":
                        (
                            100
                            * build_index
                        ),
                }
            )

    return pd.DataFrame(rows)


def split(dataframe):
    return build_split_table(
        dataframe=dataframe,
        train_fraction=0.80,
        min_train_builds=1,
        min_test_builds=1,
    )


def test_split_is_by_whole_build():
    assignments = split(
        make_data()
    )

    assert len(assignments) == 10

    assert (
        assignments[
            "split"
        ].value_counts()[
            "train"
        ]
        == 8
    )

    assert (
        assignments[
            "split"
        ].value_counts()[
            "test"
        ]
        == 2
    )


def test_train_builds_are_before_test_builds():
    assignments = split(
        make_data()
    )

    train = assignments[
        assignments["split"]
        == "train"
    ]

    test = assignments[
        assignments["split"]
        == "test"
    ]

    assert (
        train[
            "build_index"
        ].max()
        <
        test[
            "build_index"
        ].min()
    )


def test_split_is_invariant_to_row_order():
    base = make_data()

    expected = (
        split(base)
        .sort_values(
            "build_index"
        )
        .reset_index(drop=True)
    )

    shuffled = (
        base.sample(
            frac=1.0,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    actual = (
        split(shuffled)
        .sort_values(
            "build_index"
        )
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_split_does_not_depend_on_outcome_or_cost():
    base = make_data()

    expected = (
        split(base)[
            [
                "build_id",
                "split",
            ]
        ]
        .sort_values(
            "build_id"
        )
        .reset_index(drop=True)
    )

    mutated = base.copy()

    mutated["failed"] = (
        1 - mutated["failed"]
    )

    mutated[
        "execution_time_ms"
    ] = 999999

    actual = (
        split(mutated)[
            [
                "build_id",
                "split",
            ]
        ]
        .sort_values(
            "build_id"
        )
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
    )


def test_split_uses_observed_build_count_not_row_count():
    base = make_data()

    # Add many more test rows to build 1.
    extra = pd.DataFrame(
        [
            {
                "project_id": "p",
                "build_id": "1",
                "build_started_at":
                    pd.Timestamp(
                        "2026-01-02",
                        tz="UTC",
                    ),
                "build_index": 1,
                "test_id": f"X{i}",
                "failed": 0,
                "execution_time_ms": 100,
            }
            for i in range(100)
        ]
    )

    expanded = pd.concat(
        [
            base,
            extra,
        ],
        ignore_index=True,
    )

    assignments = split(
        expanded
    )

    assert len(assignments) == 10

    train_ids = set(
        assignments.loc[
            assignments["split"]
            == "train",
            "build_id",
        ]
    )

    assert train_ids == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    }