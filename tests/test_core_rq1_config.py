from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CORE_RQ1_PATH = (
    ROOT
    / "configs"
    / "core_rq1_v1.yaml"
)

RQ1_PATH = (
    ROOT
    / "configs"
    / "rq1_v1.yaml"
)

PROJECTS_PATH = (
    ROOT
    / "configs"
    / "core_projects_v1.yaml"
)


def load(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_core_rq1_identity_and_scope():
    config = load(
        CORE_RQ1_PATH
    )

    assert (
        config["execution_id"]
        == "PHASE11-CORE-RQ1-v1"
    )

    assert (
        config[
            "primary_final_rq1_evidence"
        ]
        is False
    )


def test_core_rq1_uses_frozen_project_set():
    config = load(
        CORE_RQ1_PATH
    )

    projects = load(
        PROJECTS_PATH
    )

    assert (
        config[
            "inputs"
        ][
            "project_config"
        ]
        == "configs/core_projects_v1.yaml"
    )

    assert (
        config[
            "execution"
        ][
            "project_count"
        ]
        == 10
    )

    assert len(
        projects["projects"]
    ) == 10

    assert (
        projects[
            "primary_final_rq1_evidence"
        ]
        is False
    )


def test_core_rq1_uses_core_split():
    config = load(
        CORE_RQ1_PATH
    )

    assert (
        config[
            "inputs"
        ][
            "split_root"
        ]
        ==
        "data/processed/tcp_ci/core_split"
    )

    assert (
        config[
            "references"
        ][
            "outer_split"
        ]
        ==
        "configs/core_temporal_split_v1.yaml"
    )


def test_core_rq1_preserves_frozen_rq1_method():
    core = load(
        CORE_RQ1_PATH
    )

    rq1 = load(
        RQ1_PATH
    )

    assert (
        core[
            "execution"
        ][
            "feature_set"
        ]
        ==
        rq1[
            "feature_set"
        ][
            "name"
        ]
        ==
        "F1_F2_F3"
    )

    assert (
        core[
            "execution"
        ][
            "models"
        ]
        ==
        rq1[
            "models"
        ]
    )

    assert (
        core[
            "execution"
        ][
            "calibration_variants"
        ]
        ==
        rq1[
            "calibration_variants"
        ]
    )


def test_core_rq1_grid_is_ten_by_three_by_three():
    config = load(
        CORE_RQ1_PATH
    )

    execution = (
        config[
            "execution"
        ]
    )

    expected = (
        execution[
            "project_count"
        ]
        * len(
            execution[
                "models"
            ]
        )
        * len(
            execution[
                "calibration_variants"
            ]
        )
    )

    assert (
        execution[
            "expected_conditions_per_project"
        ]
        == 9
    )

    assert (
        execution[
            "maximum_conditions"
        ]
        == expected
        == 90
    )


def test_core_rq1_is_project_specific():
    execution = load(
        CORE_RQ1_PATH
    )[
        "execution"
    ]

    assert (
        execution[
            "project_specific_models"
        ]
        is True
    )

    assert (
        execution[
            "project_specific_calibration"
        ]
        is True
    )

    assert (
        execution[
            "pooled_model_fit"
        ]
        is False
    )

    assert (
        execution[
            "pooled_calibration_fit"
        ]
        is False
    )


def test_core_rq1_result_selection_is_forbidden():
    policy = load(
        CORE_RQ1_PATH
    )[
        "result_policy"
    ]

    forbidden = [
        "select_projects_from_model_metrics",
        "select_best_project",
        "select_best_model",
        "select_best_calibration_method",
        "change_primary_feature_set",
        "pool_project_rows_as_primary_result",
        "claim_final_walk_forward_rq1_evidence",
    ]

    for field in forbidden:
        assert (
            policy[field]
            is False
        )


def test_core_rq1_phase_boundary_remains_closed():
    boundary = load(
        CORE_RQ1_PATH
    )[
        "phase_boundary"
    ]

    assert (
        boundary[
            "walk_forward_evaluation"
        ]
        is False
    )

    assert (
        boundary[
            "risk_bounded_selection"
        ]
        is False
    )

    assert (
        boundary[
            "rq3_retraining"
        ]
        is False
    )


def test_core_rq1_undefined_metric_policy_matches_rq1():
    core = load(
        CORE_RQ1_PATH
    )

    rq1 = load(
        RQ1_PATH
    )

    assert (
        core[
            "undefined_metric_policy"
        ][
            "value"
        ]
        ==
        rq1[
            "undefined_metric_policy"
        ][
            "value"
        ]
        ==
        "NA"
    )
