from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "core_project_screening_v1.yaml"
)

AMENDMENT_PATH = (
    ROOT
    / "docs"
    / "protocol_amendment_001_core_project_eligibility.md"
)


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_core_screening_identity_and_scope():
    config = load_config()

    assert (
        config["screening_id"]
        == "CORE-PROJECT-SCREENING-v1"
    )

    assert (
        config["primary_final_rq1_evidence"]
        is False
    )

    assert (
        config["project_scope"][
            "discovered_projects_expected"
        ]
        == 25
    )

    assert (
        config["project_scope"][
            "minimum_end_to_end_projects"
        ]
        == 8
    )

    assert (
        config["project_scope"][
            "target_end_to_end_projects"
        ]
        == 10
    )


def test_project_selection_is_not_performance_driven():
    config = load_config()

    policy = config[
        "selection_policy"
    ]

    assert (
        policy[
            "project_eligibility_frozen_before_model_performance"
        ]
        is True
    )

    assert (
        policy[
            "performance_based_inclusion_allowed"
        ]
        is False
    )

    assert (
        policy[
            "performance_based_exclusion_allowed"
        ]
        is False
    )

    assert (
        policy[
            "retain_frozen_pilot_anchors_if_eligible"
        ]
        is True
    )

    assert (
        policy[
            "additional_project_order"
        ][
            "field"
        ]
        == "project_key"
    )

    assert (
        policy[
            "additional_project_order"
        ][
            "order"
        ]
        == "ascending"
    )


def test_required_source_artifacts_are_frozen():
    config = load_config()

    assert config[
        "required_source_artifacts"
    ] == [
        "exe.csv",
        "builds.csv",
        "id_map.csv",
        "entity_change_history.csv",
    ]


def test_canonical_ambiguity_is_rejected():
    config = load_config()

    policy = config[
        "canonical_key_precheck"
    ]

    assert (
        policy[
            "require_zero_null_key_rows"
        ]
        is True
    )

    assert (
        policy[
            "require_zero_duplicate_build_test_groups"
        ]
        is True
    )

    assert (
        policy[
            "require_zero_multi_job_build_test_groups"
        ]
        is True
    )

    assert (
        policy[
            "require_zero_conflicting_verdict_build_test_groups"
        ]
        is True
    )


def test_current_temporal_rules_are_reused():
    config = load_config()

    temporal = config[
        "chronological_history"
    ]

    assert (
        temporal[
            "derive_boundaries_from_frozen_configs"
        ]
        is True
    )

    assert (
        temporal[
            "outer_split_reference"
        ]
        == "configs/temporal_split_v1.yaml"
    )

    assert (
        temporal[
            "rq1_reference"
        ]
        == "configs/rq1_v1.yaml"
    )

    assert (
        temporal[
            "require_initial_inner_model_fit_minimum"
        ]
        is True
    )


def test_calibration_support_does_not_select_projects():
    config = load_config()

    assert (
        config[
            "calibration_support"
        ][
            "affects_project_eligibility"
        ]
        is False
    )

    assert (
        config[
            "evaluation_class_support"
        ][
            "affects_project_eligibility"
        ]
        is False
    )


def test_performance_fields_are_forbidden():
    config = load_config()

    forbidden = set(
        config[
            "forbidden_selection_fields"
        ]
    )

    assert {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ece",
        "calibration_bias",
        "failure_recall",
        "cost_saving",
    }.issubset(
        forbidden
    )


def test_phase_boundary_excludes_later_experiments():
    config = load_config()

    boundary = config[
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


def test_protocol_amendment_exists():
    assert AMENDMENT_PATH.is_file()


def test_execution_identity_contract_is_frozen():
    config = load_config()

    identity = config[
        "execution_identity"
    ]

    assert identity[
        "staging_key_fields"
    ] == [
        "test",
        "build",
        "job",
    ]

    assert (
        identity[
            "require_zero_null_staging_key_rows"
        ]
        is True
    )

    mapping = identity[
        "test_id_mapping"
    ]

    assert (
        mapping[
            "execution_field"
        ]
        == "test"
    )

    assert (
        mapping[
            "id_map_field"
        ]
        == "value"
    )

    assert (
        mapping[
            "require_all_execution_tests_in_id_map"
        ]
        is True
    )


def test_build_metadata_identity_is_unambiguous():
    config = load_config()

    linkage = config[
        "execution_build_linkage"
    ]

    assert (
        linkage[
            "require_all_execution_build_ids_in_build_metadata"
        ]
        is True
    )

    assert (
        linkage[
            "require_zero_null_build_metadata_ids"
        ]
        is True
    )

    assert (
        linkage[
            "require_unique_build_metadata_ids"
        ]
        is True
    )

    assert (
        linkage[
            "require_all_build_timestamps_parseable"
        ]
        is True
    )
