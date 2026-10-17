from pathlib import Path

import yaml


CONFIG_ROOT = Path(
    "configs"
)


def load_yaml(
    name,
):
    with (
        CONFIG_ROOT
        / name
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def test_targeted_validation_is_not_primary_evidence():
    config = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    assert (
        config[
            "validation"
        ][
            "primary_final_evidence"
        ]
        is False
    )

    assert (
        config[
            "result_policy"
        ][
            "final_thesis_claims_allowed"
        ]
        is False
    )


def test_targeted_validation_uses_three_frozen_feature_sets():
    targeted = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    core = load_yaml(
        "core_features_v1.yaml"
    )

    expected = [
        "F1",
        "F1_F2",
        "F1_F2_F3",
    ]

    assert (
        targeted[
            "feature_ablation"
        ][
            "comparison_order"
        ]
        == expected
    )

    for name in expected:
        assert (
            name
            in core[
                "feature_sets"
            ]
        )

    assert (
        targeted[
            "feature_ablation"
        ][
            "primary_feature_set_remains"
        ]
        ==
        core[
            "primary_feature_set"
        ]
        ==
        "F1_F2_F3"
    )


def test_targeted_validation_does_not_change_rq1_models():
    targeted = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    assert (
        targeted[
            "models"
        ]
        ==
        rq1[
            "models"
        ]
        ==
        [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ]
    )


def test_targeted_validation_does_not_change_calibration_variants():
    targeted = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    assert (
        targeted[
            "calibration_variants"
        ]
        ==
        rq1[
            "calibration_variants"
        ]
        ==
        [
            "uncalibrated",
            "platt",
            "isotonic",
        ]
    )


def test_targeted_validation_preserves_temporal_isolation():
    config = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    temporal = config[
        "temporal_design"
    ]

    assert (
        temporal[
            "reuse_rq1_v1"
        ]
        is True
    )

    assert (
        temporal[
            "outer_evaluation_boundary_unchanged"
        ]
        is True
    )

    assert (
        temporal[
            "calibration_window_rule_unchanged"
        ]
        is True
    )

    assert (
        temporal[
            "minimum_support_rule_unchanged"
        ]
        is True
    )

    assert (
        temporal[
            "evaluation_used_for_model_fit"
        ]
        is False
    )

    assert (
        temporal[
            "evaluation_used_for_calibration_fit"
        ]
        is False
    )

    assert (
        temporal[
            "evaluation_used_for_feature_selection"
        ]
        is False
    )


def test_targeted_validation_has_no_result_driven_selection():
    config = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    assert (
        config[
            "feature_ablation"
        ][
            "feature_selection_from_results"
        ]
        is False
    )

    policy = config[
        "result_policy"
    ]

    assert (
        policy[
            "select_best_feature_set"
        ]
        is False
    )

    assert (
        policy[
            "select_best_model"
        ]
        is False
    )

    assert (
        policy[
            "select_best_calibration_method"
        ]
        is False
    )

    assert (
        policy[
            "change_primary_feature_set_from_results"
        ]
        is False
    )


def test_targeted_validation_expected_grid_is_frozen():
    config = load_yaml(
        "rq1_targeted_validation_v1.yaml"
    )

    grid = config[
        "expected_grid"
    ]

    assert grid[
        "projects"
    ] == 5

    assert grid[
        "feature_sets"
    ] == 3

    assert grid[
        "models"
    ] == 3

    assert grid[
        "calibration_variants"
    ] == 3

    assert (
        grid[
            "projects"
        ]
        *
        grid[
            "feature_sets"
        ]
        *
        grid[
            "models"
        ]
        *
        grid[
            "calibration_variants"
        ]
        ==
        grid[
            "maximum_conditions"
        ]
        ==
        135
    )
