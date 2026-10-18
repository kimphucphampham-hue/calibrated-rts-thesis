from pathlib import Path

import pandas as pd
import yaml

from src.tcp_ci.screen_core_projects import (
    calculate_temporal_structure,
    discover_project_dirs,
    screen_project,
    validate_no_performance_fields,
)


def screening_config():
    return {
        "required_source_artifacts": [
            "exe.csv",
            "builds.csv",
            "id_map.csv",
            "entity_change_history.csv",
        ],

        "target_validity": {
            "allowed_values": [
                0,
                1,
                2,
                3,
            ],
        },

        "forbidden_selection_fields": [
            "roc_auc",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece",
        ],
    }


def temporal_config():
    return {
        "split": {
            "train_fraction": 0.80,
            "min_train_builds": 100,
            "min_test_builds": 20,
        }
    }


def rq1_config():
    return {
        "temporal_design": {
            "inner_initial": {
                "model_fit_fraction_of_development":
                    0.80,
            },

            "calibration_expansion": {
                "minimum_model_fit_builds":
                    100,
            },
        }
    }


def write_project(
    root: Path,
    *,
    project_key: str =
        "datasets/example@repo",
    duplicate_build_test: bool =
        False,
    unmapped_test: bool =
        False,
    single_class_fit: bool =
        False,
):
    project_dir = (
        root
        / project_key
    )

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index in range(
        1,
        161,
    ):
        if single_class_fit:
            verdict = (
                0
                if index <= 100
                else 1
            )

        else:
            verdict = (
                1
                if index % 20 == 0
                else 0
            )

        rows.append(
            {
                "test":
                    (
                        "999"
                        if (
                            unmapped_test
                            and
                            index == 1
                        )
                        else
                        "1"
                    ),

                "build":
                    str(
                        index
                    ),

                "job":
                    "1",

                "verdict":
                    verdict,

                "duration":
                    10,
            }
        )

    if duplicate_build_test:
        rows.append(
            dict(
                rows[
                    0
                ]
            )
        )

    pd.DataFrame(
        rows
    ).to_csv(
        project_dir
        / "exe.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "id": [
                str(
                    value
                )
                for value
                in range(
                    1,
                    161,
                )
            ],

            "commits": [
                ""
                for _
                in range(
                    160
                )
            ],

            "started_at":
                pd.date_range(
                    "2020-01-01",
                    periods=160,
                    freq="D",
                    tz="UTC",
                ).astype(
                    str
                ),
        }
    ).to_csv(
        project_dir
        / "builds.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "key": [
                "ExampleTest",
            ],
            "value": [
                "1",
            ],
        }
    ).to_csv(
        project_dir
        / "id_map.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Commit": [],
            "EntityId": [],
            "CommitDate": [],
        }
    ).to_csv(
        project_dir
        / "entity_change_history.csv",
        index=False,
    )

    return project_dir


def run_project(
    project_dir,
):
    return screen_project(
        project_key=
            "datasets/example@repo",
        project_dir=
            project_dir,
        config=
            screening_config(),
        temporal_config=
            temporal_config(),
        rq1_config=
            rq1_config(),
        pilot_projects=
            set(),
    )


def test_temporal_structure_matches_frozen_math():
    result = (
        calculate_temporal_structure(
            [
                str(
                    value
                )
                for value
                in range(
                    1,
                    161,
                )
            ],
            temporal_config(),
            rq1_config(),
        )
    )

    assert result[
        "observed_builds"
    ] == 160

    assert result[
        "development_builds"
    ] == 128

    assert result[
        "evaluation_builds"
    ] == 32

    assert result[
        "initial_model_fit_builds"
    ] == 102

    assert result[
        "initial_calibration_builds"
    ] == 26


def test_structurally_valid_project_is_eligible(
    tmp_path,
):
    project_dir = write_project(
        tmp_path
    )

    row = run_project(
        project_dir
    )

    assert (
        row[
            "eligible_phase11"
        ]
        is True
    )

    assert (
        row[
            "exclusion_reasons"
        ]
        == ""
    )

    assert (
        row[
            "model_fit_both_classes"
        ]
        is True
    )


def test_duplicate_build_test_group_is_rejected(
    tmp_path,
):
    project_dir = write_project(
        tmp_path,
        duplicate_build_test=True,
    )

    row = run_project(
        project_dir
    )

    assert (
        row[
            "eligible_phase11"
        ]
        is False
    )

    assert (
        row[
            "duplicate_build_test_groups"
        ]
        == 1
    )

    assert (
        "CANONICAL_KEY_AMBIGUITY"
        in row[
            "exclusion_reasons"
        ]
    )


def test_unmapped_test_identifier_is_rejected(
    tmp_path,
):
    project_dir = write_project(
        tmp_path,
        unmapped_test=True,
    )

    row = run_project(
        project_dir
    )

    assert (
        row[
            "eligible_phase11"
        ]
        is False
    )

    assert (
        row[
            "unmapped_test_ids"
        ]
        == 1
    )

    assert (
        "UNMAPPED_TEST_ID"
        in row[
            "exclusion_reasons"
        ]
    )


def test_single_class_model_fit_is_rejected(
    tmp_path,
):
    project_dir = write_project(
        tmp_path,
        single_class_fit=True,
    )

    row = run_project(
        project_dir
    )

    assert (
        row[
            "initial_model_fit_builds"
        ]
        == 102
    )

    assert (
        row[
            "guaranteed_model_fit_builds"
        ]
        == 100
    )

    assert (
        row[
            "eligible_phase11"
        ]
        is False
    )

    assert (
        row[
            "model_fit_both_classes"
        ]
        is False
    )

    assert (
        "MODEL_FIT_SINGLE_CLASS"
        in row[
            "exclusion_reasons"
        ]
    )


def test_discovery_is_deterministic(
    tmp_path,
):
    write_project(
        tmp_path,
        project_key=
            "datasets/z@z",
    )

    write_project(
        tmp_path,
        project_key=
            "datasets/a@a",
    )

    projects = discover_project_dirs(
        tmp_path
    )

    assert [
        project_key
        for (
            project_key,
            _,
        )
        in projects
    ] == [
        "datasets/a@a",
        "datasets/z@z",
    ]


def test_performance_columns_are_forbidden():
    frame = pd.DataFrame(
        {
            "project_key": [
                "x",
            ],
            "roc_auc": [
                0.9,
            ],
        }
    )

    try:
        validate_no_performance_fields(
            frame,
            screening_config(),
        )

    except AssertionError:
        pass

    else:
        raise AssertionError(
            "Expected performance-field guard."
        )


def test_structural_output_does_not_need_performance():
    frame = pd.DataFrame(
        {
            "project_key": [
                "x",
            ],

            "eligible_phase11": [
                True,
            ],

            "observed_builds": [
                160,
            ],
        }
    )

    validate_no_performance_fields(
        frame,
        screening_config(),
    )
