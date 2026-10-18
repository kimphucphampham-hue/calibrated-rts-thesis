from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(relative_path):
    path = (
        ROOT
        / relative_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def test_core_adapter_paths_are_isolated():
    config = load(
        "configs/core_adapter_v1.yaml"
    )

    assert (
        config[
            "inputs"
        ][
            "pilot_projects_config"
        ]
        == "configs/core_projects_v1.yaml"
    )

    assert (
        config[
            "outputs"
        ][
            "interim_root"
        ]
        == "data/interim/tcp_ci/core"
    )

    assert (
        config[
            "outputs"
        ][
            "processed_root"
        ]
        == "data/processed/tcp_ci/core"
    )

    assert (
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
        == "results/core/adapter"
    )

    assert (
        config[
            "duration"
        ][
            "source_unit"
        ]
        == "milliseconds"
    )

    assert (
        config[
            "duration"
        ][
            "canonical_conversion"
        ]
        == "identity"
    )

    assert (
        config[
            "canonical_target"
        ][
            "status"
        ]
        == "CONFIRMED_FOR_PHASE11_V1"
    )

    notes = " ".join(
        config[
            "notes"
        ]
    )

    assert (
        "not yet been verified"
        not in notes
    )


def test_core_history_paths_are_isolated():
    config = load(
        "configs/core_history_features_v1.yaml"
    )

    assert (
        config[
            "inputs"
        ][
            "canonical_root"
        ]
        == "data/processed/tcp_ci/core"
    )

    assert (
        config[
            "inputs"
        ][
            "pilot_projects_config"
        ]
        == "configs/core_projects_v1.yaml"
    )

    assert (
        config[
            "outputs"
        ][
            "feature_root"
        ]
        == "data/processed/tcp_ci/core_history"
    )

    assert (
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
        == "results/core/history_features"
    )


def test_core_split_paths_are_isolated():
    config = load(
        "configs/core_temporal_split_v1.yaml"
    )

    assert (
        config[
            "inputs"
        ][
            "history_root"
        ]
        == "data/processed/tcp_ci/core_history"
    )

    assert (
        config[
            "inputs"
        ][
            "pilot_projects_config"
        ]
        == "configs/core_projects_v1.yaml"
    )

    assert (
        config[
            "outputs"
        ][
            "split_root"
        ]
        == "data/processed/tcp_ci/core_split"
    )

    assert (
        config[
            "outputs"
        ][
            "diagnostics_root"
        ]
        == "results/core/temporal_split"
    )

    notes = " ".join(
        config[
            "notes"
        ]
    )

    assert (
        "Phase 11"
        in notes
    )

    assert (
        "initial pilot validation"
        not in notes
    )


def normalize_adapter(
    config,
):
    result = deepcopy(
        config
    )

    # Scope text may differ.
    result[
        "purpose"
    ] = "__NORMALIZED__"

    result[
        "inputs"
    ][
        "pilot_projects_config"
    ] = "__PROJECT_CONFIG__"

    result[
        "outputs"
    ][
        "interim_root"
    ] = "__INTERIM__"

    result[
        "outputs"
    ][
        "processed_root"
    ] = "__PROCESSED__"

    result[
        "outputs"
    ][
        "diagnostics_root"
    ] = "__DIAGNOSTICS__"

    result[
        "canonical_target"
    ][
        "status"
    ] = "__STATUS__"

    result[
        "notes"
    ] = "__NOTES__"

    return result


def test_core_adapter_preserves_pilot_methodology():
    pilot = load(
        "configs/tcp_ci_adapter_v1.yaml"
    )

    core = load(
        "configs/core_adapter_v1.yaml"
    )

    assert (
        normalize_adapter(
            core
        )
        == normalize_adapter(
            pilot
        )
    )


def normalize_history(
    config,
):
    result = deepcopy(
        config
    )

    result[
        "purpose"
    ] = "__NORMALIZED__"

    result[
        "inputs"
    ][
        "canonical_root"
    ] = "__CANONICAL__"

    result[
        "inputs"
    ][
        "pilot_projects_config"
    ] = "__PROJECT_CONFIG__"

    result[
        "outputs"
    ][
        "feature_root"
    ] = "__FEATURE_ROOT__"

    result[
        "outputs"
    ][
        "diagnostics_root"
    ] = "__DIAGNOSTICS__"

    return result


def test_core_history_preserves_frozen_feature_methodology():
    pilot = load(
        "configs/history_features_v1.yaml"
    )

    core = load(
        "configs/core_history_features_v1.yaml"
    )

    assert (
        normalize_history(
            core
        )
        == normalize_history(
            pilot
        )
    )


def normalize_split(
    config,
):
    result = deepcopy(
        config
    )

    result[
        "purpose"
    ] = "__NORMALIZED__"

    result[
        "inputs"
    ][
        "history_root"
    ] = "__HISTORY__"

    result[
        "inputs"
    ][
        "pilot_projects_config"
    ] = "__PROJECT_CONFIG__"

    result[
        "outputs"
    ][
        "split_root"
    ] = "__SPLIT_ROOT__"

    result[
        "outputs"
    ][
        "diagnostics_root"
    ] = "__DIAGNOSTICS__"

    result[
        "notes"
    ] = "__NOTES__"

    return result


def test_core_split_preserves_frozen_temporal_methodology():
    pilot = load(
        "configs/temporal_split_v1.yaml"
    )

    core = load(
        "configs/core_temporal_split_v1.yaml"
    )

    assert (
        normalize_split(
            core
        )
        == normalize_split(
            pilot
        )
    )
