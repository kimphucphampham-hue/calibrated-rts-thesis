from pathlib import Path

import yaml


PROTOCOL_PATH = Path(
    "configs/protocol_v1.yaml"
)


def load_protocol():
    with PROTOCOL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_protocol_file_exists():
    assert PROTOCOL_PATH.is_file()


def test_referenced_configs_exist():
    protocol = load_protocol()

    for path in protocol[
        "references"
    ].values():
        assert Path(path).is_file(), path


def test_primary_temporal_protocol_is_walk_forward():
    protocol = load_protocol()

    core = protocol[
        "core_evaluation"
    ]

    assert (
        core["strategy"]
        == "walk_forward"
    )

    assert (
        core[
            "random_k_fold_allowed"
        ]
        is False
    )


def test_pilot_and_core_evaluation_are_distinct():
    protocol = load_protocol()

    assert (
        protocol[
            "pilot_evaluation"
        ][
            "role"
        ]
        == "pipeline validation only"
    )

    assert (
        protocol[
            "pilot_evaluation"
        ][
            "strategy"
        ]
        != protocol[
            "core_evaluation"
        ][
            "strategy"
        ]
    )


def test_current_outcome_and_cost_are_not_features():
    protocol = load_protocol()

    leakage = protocol[
        "leakage"
    ]

    assert (
        leakage[
            "current_failed_as_feature"
        ]
        is False
    )

    assert (
        leakage[
            "current_execution_time_as_feature"
        ]
        is False
    )


def test_calibration_methods_are_frozen():
    protocol = load_protocol()

    assert protocol[
        "calibration"
    ][
        "methods"
    ] == [
        "uncalibrated",
        "platt",
        "isotonic",
    ]

    assert (
        protocol[
            "calibration"
        ][
            "test_data_used_for_calibrator_fit"
        ]
        is False
    )


def test_risk_levels_are_frozen():
    protocol = load_protocol()

    assert protocol[
        "risk_bounded_selection"
    ][
        "rho"
    ] == [
        0.90,
        0.95,
        0.99,
    ]


def test_project_selection_is_not_performance_based():
    protocol = load_protocol()

    assert (
        protocol[
            "project_selection"
        ][
            "selection_must_be_independent_of_model_performance"
        ]
        is True
    )


def test_core_project_scope():
    protocol = load_protocol()

    scope = protocol[
        "scope"
    ]

    assert (
        scope[
            "core_minimum_projects"
        ]
        >= 8
    )

    assert (
        scope[
            "core_target_projects"
        ]
        >= scope[
            "core_minimum_projects"
        ]
    )


def test_protocol_amendments_are_versioned():
    protocol = load_protocol()

    assert (
        protocol[
            "protocol_change"
        ][
            "require_version_increment_for_methodological_change"
        ]
        is True
    )