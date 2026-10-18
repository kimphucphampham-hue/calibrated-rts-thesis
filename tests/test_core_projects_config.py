from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "core_projects_v1.yaml"
)

PILOT_PATH = (
    ROOT
    / "configs"
    / "pilot_projects_v1.yaml"
)


EXPECTED_ADDITIONAL = [
    "datasets/CompEvol@beast2",
    "datasets/EMResearch@EvoMaster",
    "datasets/Graylog2@graylog2-server",
    "datasets/JMRI@JMRI",
    "datasets/SonarSource@sonarqube",
]


def load_yaml(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def test_core_project_set_identity():
    config = load_yaml(
        CONFIG_PATH
    )

    assert (
        config[
            "project_set_id"
        ]
        == "PHASE11-CORE-PROJECTS-v1"
    )

    assert (
        config[
            "primary_final_rq1_evidence"
        ]
        is False
    )


def test_exactly_ten_unique_projects_are_frozen():
    config = load_yaml(
        CONFIG_PATH
    )

    projects = config[
        "projects"
    ]

    keys = [
        item[
            "project_key"
        ]
        for item
        in projects
    ]

    assert len(
        keys
    ) == 10

    assert len(
        set(
            keys
        )
    ) == 10


def test_all_five_pilot_anchors_are_retained():
    config = load_yaml(
        CONFIG_PATH
    )

    pilot = load_yaml(
        PILOT_PATH
    )

    expected_pilots = {
        item[
            "project_key"
        ]
        for item
        in pilot[
            "projects"
        ]
    }

    actual_pilots = {
        item[
            "project_key"
        ]
        for item
        in config[
            "projects"
        ]
        if item[
            "selection_role"
        ]
        == "pilot_anchor"
    }

    assert len(
        actual_pilots
    ) == 5

    assert (
        actual_pilots
        == expected_pilots
    )


def test_additional_projects_are_exact_and_ordered():
    config = load_yaml(
        CONFIG_PATH
    )

    additional = [
        item[
            "project_key"
        ]
        for item
        in config[
            "projects"
        ]
        if item[
            "selection_role"
        ]
        == "deterministic_additional"
    ]

    assert (
        additional
        == EXPECTED_ADDITIONAL
    )

    assert (
        additional
        == sorted(
            additional
        )
    )


def test_project_selection_did_not_use_performance():
    config = load_yaml(
        CONFIG_PATH
    )

    policy = config[
        "selection_policy"
    ]

    assert (
        policy[
            "model_performance_used"
        ]
        is False
    )

    assert (
        policy[
            "pilot_anchors_retained"
        ]
        is True
    )

    assert (
        policy[
            "additional_project_order"
        ]
        == "project_key_ascending"
    )

    assert (
        policy[
            "stop_when_target_reached"
        ]
        is True
    )

    assert (
        policy[
            "eligible_not_selected_are_ineligible"
        ]
        is False
    )


def test_screening_provenance_is_frozen():
    config = load_yaml(
        CONFIG_PATH
    )

    provenance = config[
        "screening_provenance"
    ]

    assert (
        provenance[
            "git_commit"
        ]
        == (
            "f1a20a18abdfc5ce8f2e4fe2fd0abf"
            "76a1a28df5"
        )
    )

    assert (
        provenance[
            "screening_sha256"
        ]
        == (
            "6d2ffb5ec4d1da77c80e8745b235652"
            "f3d3714a3e5fe31330ac9e28bab3679bb"
        )
    )

    assert (
        provenance[
            "screened_projects"
        ]
        == 25
    )

    assert (
        provenance[
            "structurally_eligible_projects"
        ]
        == 25
    )


def test_phase_boundary_remains_non_final():
    config = load_yaml(
        CONFIG_PATH
    )

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
