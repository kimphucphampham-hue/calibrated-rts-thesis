from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.tcp_ci.calibration_split import (
    calibration_support,
    materialize_calibration_partitions,
    split_development_for_calibration,
)


CONFIG_DIR = Path("configs")


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_configs() -> tuple[dict, dict]:
    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    return (
        deepcopy(rq1),
        deepcopy(calibration),
    )


def make_development(
    builds,
    failed_rows_by_build=None,
) -> pd.DataFrame:
    if failed_rows_by_build is None:
        failed_rows_by_build = {}

    rows = []

    for build_index in builds:
        # Keep at least one negative row
        # in every build.
        rows.append(
            {
                "build_index":
                    build_index,
                "failed":
                    0,
            }
        )

        failed_rows = int(
            failed_rows_by_build.get(
                build_index,
                0,
            )
        )

        for _ in range(
            failed_rows
        ):
            rows.append(
                {
                    "build_index":
                        build_index,
                    "failed":
                        1,
                }
            )

    return pd.DataFrame(rows)


def test_calibration_support_counts_rows_and_builds():
    data = make_development(
        builds=[
            1,
            2,
            3,
            4,
        ],
        failed_rows_by_build={
            2: 2,
            3: 3,
        },
    )

    support = calibration_support(
        data
    )

    assert (
        support.positive_rows
        == 5
    )

    assert (
        support.negative_rows
        == 4
    )

    assert (
        support.failing_builds
        == 2
    )

    assert (
        support.passing_builds
        == 2
    )


def test_initial_window_is_used_when_support_is_sufficient():
    rq1, calibration = (
        load_configs()
    )

    data = make_development(
        builds=range(
            1,
            131,
        ),
        failed_rows_by_build={
            120: 3,
            129: 2,
        },
    )

    split = (
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )
    )

    assert (
        split.initial_model_fit_builds
        == 104
    )

    assert (
        len(
            split.model_fit_builds
        )
        == 104
    )

    assert (
        len(
            split.calibration_builds
        )
        == 26
    )

    assert (
        split.expanded_by_builds
        == 0
    )

    assert split.sufficient is True

    assert (
        split.status
        == "SUFFICIENT"
    )

    assert (
        split.model_fit_builds[-1]
        == 104
    )

    assert (
        split.calibration_builds[0]
        == 105
    )


def test_backward_expansion_stops_at_first_sufficient_window():
    rq1, calibration = (
        load_configs()
    )

    data = make_development(
        builds=range(
            1,
            141,
        ),
        failed_rows_by_build={
            # Initial calibration:
            # builds 113..140.
            # Four positive rows only.
            120: 1,
            125: 1,
            130: 1,
            135: 1,

            # After seven backward steps,
            # calibration starts at build 106.
            # This becomes the fifth positive.
            106: 1,

            # Build 105 would also add a
            # failure, but must NOT enter
            # calibration because support
            # became sufficient at build 106.
            105: 1,
        },
    )

    split = (
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )
    )

    assert (
        split.initial_model_fit_builds
        == 112
    )

    assert (
        split.expanded_by_builds
        == 7
    )

    assert (
        len(
            split.model_fit_builds
        )
        == 105
    )

    assert (
        len(
            split.calibration_builds
        )
        == 35
    )

    assert (
        split.model_fit_builds[-1]
        == 105
    )

    assert (
        split.calibration_builds[0]
        == 106
    )

    assert (
        split.support.positive_rows
        == 5
    )

    assert (
        split.support.failing_builds
        == 5
    )

    assert split.sufficient is True

    assert (
        split.status
        == "SUFFICIENT"
    )


def test_insufficient_support_stops_at_model_fit_floor():
    rq1, calibration = (
        load_configs()
    )

    data = make_development(
        builds=range(
            1,
            131,
        )
    )

    split = (
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )
    )

    # Initial floor:
    # floor(130 * 0.80) = 104.
    assert (
        split.initial_model_fit_builds
        == 104
    )

    # Expansion may move 104 -> 103
    # -> 102 -> 101 -> 100, but never
    # below 100.
    assert (
        len(
            split.model_fit_builds
        )
        == 100
    )

    assert (
        len(
            split.calibration_builds
        )
        == 30
    )

    assert (
        split.expanded_by_builds
        == 4
    )

    assert split.sufficient is False

    assert (
        split.status
        ==
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    )

    assert (
        split.support.positive_rows
        == 0
    )

    assert (
        split.support.failing_builds
        == 0
    )


def test_non_contiguous_build_indices_preserve_chronology():
    rq1, calibration = (
        load_configs()
    )

    builds = [
        value * 10
        for value in range(
            1,
            131,
        )
    ]

    data = make_development(
        builds=builds,
        failed_rows_by_build={
            1200: 3,
            1290: 2,
        },
    )

    split = (
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )
    )

    assert (
        len(
            split.model_fit_builds
        )
        == 104
    )

    assert (
        len(
            split.calibration_builds
        )
        == 26
    )

    assert (
        split.model_fit_builds[-1]
        == 1040
    )

    assert (
        split.calibration_builds[0]
        == 1050
    )

    assert (
        max(
            split.model_fit_builds
        )
        <
        min(
            split.calibration_builds
        )
    )


def test_invalid_non_binary_target_is_rejected():
    rq1, calibration = (
        load_configs()
    )

    data = make_development(
        builds=range(
            1,
            131,
        )
    )

    data.loc[
        data.index[-1],
        "failed",
    ] = 2

    with pytest.raises(
        ValueError,
        match="binary 0/1",
    ):
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )


def test_history_too_short_for_model_fit_floor_is_rejected():
    rq1, calibration = (
        load_configs()
    )

    # floor(124 * 0.80) = 99,
    # below the frozen 100-build floor.
    data = make_development(
        builds=range(
            1,
            125,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "below the configured "
            "minimum_model_fit_builds"
        ),
    ):
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration,
        )


def test_materialized_partitions_are_complete_and_disjoint():
    rq1, calibration_config = (
        load_configs()
    )

    data = make_development(
        builds=range(
            1,
            131,
        ),
        failed_rows_by_build={
            120: 3,
            129: 2,
        },
    )

    split = (
        split_development_for_calibration(
            data,
            rq1_config=rq1,
            calibration_config=
                calibration_config,
        )
    )

    model_fit, calibration = (
        materialize_calibration_partitions(
            data,
            split,
        )
    )

    fit_builds = set(
        model_fit[
            "build_index"
        ].unique()
    )

    calibration_builds = set(
        calibration[
            "build_index"
        ].unique()
    )

    assert fit_builds == set(
        split.model_fit_builds
    )

    assert calibration_builds == set(
        split.calibration_builds
    )

    assert fit_builds.isdisjoint(
        calibration_builds
    )

    assert (
        max(fit_builds)
        <
        min(calibration_builds)
    )

    assert (
        len(model_fit)
        + len(calibration)
        == len(data)
    )
