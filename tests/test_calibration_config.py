from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path("configs")


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_calibration_methods_match_rq1_variants():
    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    assert calibration["methods"] == (
        rq1["calibration_variants"]
    )

    assert calibration["methods"] == [
        "uncalibrated",
        "platt",
        "isotonic",
    ]


def test_outer_temporal_rules_are_consistent():
    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    outer = load_yaml(
        "temporal_split_v1.yaml"
    )

    rq1_outer = (
        rq1["temporal_design"]["outer"]
    )

    assert (
        rq1_outer[
            "development_fraction"
        ]
        ==
        outer["split"]["train_fraction"]
    )

    assert (
        rq1_outer[
            "evaluation_fraction"
        ]
        ==
        pytest.approx(
            1.0
            - outer[
                "split"
            ][
                "train_fraction"
            ]
        )
    )

    assert (
        rq1_outer[
            "rounding"
        ]
        ==
        outer["split"]["rounding"]
    )

    assert (
        outer[
            "temporal_rule"
        ][
            "train_before_test"
        ]
        is True
    )

    assert (
        outer[
            "holdout_policy"
        ][
            "test_used_for_model_fit"
        ]
        is False
    )

    assert (
        outer[
            "holdout_policy"
        ][
            "test_used_for_calibration_fit"
        ]
        is False
    )


def test_inner_temporal_design_is_frozen():
    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    temporal = rq1[
        "temporal_design"
    ]

    inner = temporal[
        "inner_initial"
    ]

    expansion = temporal[
        "calibration_expansion"
    ]

    assert (
        inner[
            "model_fit_fraction_of_development"
        ]
        == pytest.approx(0.80)
    )

    assert (
        inner[
            "calibration_fraction_of_development"
        ]
        == pytest.approx(0.20)
    )

    assert (
        inner[
            "model_fit_fraction_of_development"
        ]
        +
        inner[
            "calibration_fraction_of_development"
        ]
        == pytest.approx(1.0)
    )

    assert (
        inner["rounding"]
        == "floor"
    )

    assert (
        expansion["enabled"]
        is True
    )

    assert (
        expansion["direction"]
        == "backward_in_time"
    )

    assert (
        expansion["step_unit"]
        == "one_build"
    )

    assert (
        expansion[
            "stop_at_first_sufficient_window"
        ]
        is True
    )

    assert (
        expansion[
            "evaluation_builds_may_be_used"
        ]
        is False
    )

    assert (
        expansion[
            "minimum_model_fit_builds"
        ]
        == 100
    )


def test_calibration_support_rules_match_rq1():
    rq1 = load_yaml(
        "rq1_v1.yaml"
    )

    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    rq1_expansion = (
        rq1[
            "temporal_design"
        ][
            "calibration_expansion"
        ]
    )

    rq1_support = (
        rq1_expansion[
            "minimum_support"
        ]
    )

    calibration_policy = (
        calibration[
            "rare_event_policy"
        ]
    )

    calibration_support = (
        calibration_policy[
            "minimum_support"
        ]
    )

    assert (
        rq1_support
        ==
        calibration_support
    )

    assert rq1_support == {
        "positive_rows": 5,
        "negative_rows": 5,
        "failing_builds": 2,
        "passing_builds": 2,
    }

    assert (
        rq1_expansion[
            "minimum_model_fit_builds"
        ]
        ==
        calibration_policy[
            "minimum_model_fit_builds"
        ]
        ==
        100
    )

    assert (
        calibration_policy[
            "expansion"
        ][
            "enabled"
        ]
        is True
    )

    assert (
        calibration_policy[
            "expansion"
        ][
            "direction"
        ]
        == "backward"
    )

    assert (
        calibration_policy[
            "expansion"
        ][
            "step"
        ]
        == "one_build"
    )

    assert (
        calibration_policy[
            "outer_evaluation_may_be_consumed"
        ]
        is False
    )


def test_calibration_scope_and_chronology_are_frozen():
    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    chronology = calibration[
        "chronology"
    ]

    scope = calibration[
        "calibration_scope"
    ]

    assert (
        chronology[
            "required_order"
        ]
        == [
            "model_fit",
            "calibration",
            "evaluation",
        ]
    )

    assert (
        chronology[
            "evaluation_labels_used_for_calibration_fit"
        ]
        is False
    )

    assert (
        chronology[
            "evaluation_predictions_used_for_calibration_fit"
        ]
        is False
    )

    assert (
        scope["project_specific"]
        is True
    )

    assert (
        scope["model_specific"]
        is True
    )

    assert (
        scope["shared_across_projects"]
        is False
    )

    assert (
        scope["shared_across_models"]
        is False
    )


def test_platt_and_isotonic_design_is_frozen():
    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    platt = calibration["platt"]
    isotonic = calibration[
        "isotonic"
    ]

    assert (
        platt["input"]
        == "base_probability"
    )

    assert (
        platt["epsilon"]
        == pytest.approx(
            0.000001
        )
    )

    assert (
        platt[
            "hyperparameter_tuning"
        ]
        is False
    )

    assert (
        isotonic["input"]
        == "base_probability"
    )

    assert (
        isotonic[
            "monotonic_direction"
        ]
        == "increasing"
    )

    assert (
        isotonic[
            "out_of_bounds"
        ]
        == "clip"
    )


def test_insufficient_support_policy_is_frozen():
    calibration = load_yaml(
        "calibration_v1.yaml"
    )

    policy = calibration[
        "rare_event_policy"
    ]

    action = policy[
        "insufficient_support_action"
    ]

    assert (
        action["uncalibrated"]
        == "run"
    )

    assert (
        action["platt"]
        == "skip_with_status"
    )

    assert (
        action["isotonic"]
        == "skip_with_status"
    )

    assert (
        policy["skip_status"]
        ==
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    )
