from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.tcp_ci.rq1_pipeline import (
    run_rq1_model,
)
from src.tcp_ci.rq1_targeted_validation import (
    TARGETED_FEATURE_SET_ORDER,
    run_targeted_rq1_model,
    targeted_feature_columns,
    validate_targeted_validation_contract,
)


CONFIG_DIR = Path(
    "configs"
)


def load_yaml(
    name,
):
    with (
        CONFIG_DIR
        / name
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def load_configs():
    return (
        load_yaml(
            "rq1_targeted_validation_v1.yaml"
        ),
        load_yaml(
            "rq1_v1.yaml"
        ),
        load_yaml(
            "rq1_models_v1.yaml"
        ),
        load_yaml(
            "calibration_v1.yaml"
        ),
        load_yaml(
            "core_features_v1.yaml"
        ),
    )


def make_partition(
    builds,
    failure_builds,
):
    rows = []

    for build in builds:
        failed = int(
            build
            in failure_builds
        )

        rows.append(
            {
                "project_id":
                    "project-a",

                "build_id":
                    f"build-{build}",

                "build_index":
                    build,

                "test_id":
                    "test-a",

                "failed":
                    failed,

                "failure_rate_history":
                    (build % 7)
                    / 10.0,

                "recent_failure":
                    int(
                        (
                            build - 1
                        )
                        in failure_builds
                    ),

                "consecutive_failures":
                    build % 2,

                "consecutive_passes":
                    build % 5,

                "test_age_builds":
                    build,

                "time_since_last_failure_builds":
                    (
                        build % 11
                    )
                    + 1,

                "execution_time_mean_history_ms":
                    100.0
                    + build,

                "changed_entity_count":
                    build % 5,

                "recent_failure_density_10":
                    (
                        build % 4
                    )
                    / 10.0,

                "execution_time_trend_10":
                    float(
                        (
                            build % 6
                        )
                        - 3
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_sufficient_data():
    development_failures = {
        10,
        30,
        50,
        70,
        90,
        106,
        110,
        114,
        118,
        122,
        126,
    }

    evaluation_failures = {
        132,
        134,
    }

    development = (
        make_partition(
            builds=range(
                1,
                131,
            ),
            failure_builds=
                development_failures,
        )
    )

    evaluation = (
        make_partition(
            builds=range(
                131,
                136,
            ),
            failure_builds=
                evaluation_failures,
        )
    )

    return (
        development,
        evaluation,
    )


def test_targeted_contract_resolves_three_frozen_feature_sets():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    resolved = (
        validate_targeted_validation_contract(
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    assert tuple(
        resolved.keys()
    ) == TARGETED_FEATURE_SET_ORDER

    assert [
        len(
            resolved[
                name
            ]
        )
        for name in TARGETED_FEATURE_SET_ORDER
    ] == [
        7,
        8,
        11,
    ]


@pytest.mark.parametrize(
    (
        "feature_set_name",
        "expected_count",
    ),
    [
        (
            "F1",
            7,
        ),
        (
            "F1_F2",
            8,
        ),
        (
            "F1_F2_F3",
            11,
        ),
    ],
)
def test_targeted_feature_columns_follow_frozen_sets(
    feature_set_name,
    expected_count,
):
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    features = (
        targeted_feature_columns(
            feature_set_name=
                feature_set_name,
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    assert len(
        features
    ) == expected_count

    assert features == (
        core[
            "feature_sets"
        ][
            feature_set_name
        ][
            "features"
        ]
    )


def test_unknown_targeted_feature_set_is_rejected():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    with pytest.raises(
        ValueError,
        match="Unsupported targeted feature set",
    ):
        targeted_feature_columns(
            feature_set_name=
                "F_UNKNOWN",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ],
)
def test_full_core_targeted_runner_matches_frozen_rq1_runner(
    model_name,
):
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    (
        frozen_predictions,
        frozen_diagnostics,
    ) = run_rq1_model(
        development=development,
        evaluation=evaluation,
        model_name=
            model_name,
        rq1_config=
            rq1,
        model_config=
            models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    (
        targeted_predictions,
        targeted_diagnostics,
    ) = run_targeted_rq1_model(
        development=development,
        evaluation=evaluation,
        feature_set_name=
            "F1_F2_F3",
        model_name=
            model_name,
        targeted_config=
            targeted,
        rq1_config=
            rq1,
        model_config=
            models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    pd.testing.assert_frame_equal(
        targeted_predictions,
        frozen_predictions,
        check_exact=True,
    )

    assert (
        targeted_diagnostics[
            "feature_set"
        ]
        ==
        "F1_F2_F3"
    )

    targeted_without_feature_set = {
        key:
            value
        for (
            key,
            value,
        ) in (
            targeted_diagnostics.items()
        )
        if key != "feature_set"
    }

    assert (
        targeted_without_feature_set
        ==
        frozen_diagnostics
    )


def test_targeted_runner_does_not_mutate_frozen_configs():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    before = deepcopy(
        (
            targeted,
            rq1,
            models,
            calibration,
            core,
        )
    )

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    run_targeted_rq1_model(
        development=development,
        evaluation=evaluation,
        feature_set_name=
            "F1",
        model_name=
            "logistic_regression",
        targeted_config=
            targeted,
        rq1_config=
            rq1,
        model_config=
            models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    after = (
        targeted,
        rq1,
        models,
        calibration,
        core,
    )

    assert after == before


@pytest.mark.parametrize(
    "field",
    [
        "reuse_rq1_v1",
        "outer_evaluation_boundary_unchanged",
        "calibration_window_rule_unchanged",
        "minimum_support_rule_unchanged",
    ],
)
def test_contract_rejects_temporal_guard_drift(
    field,
):
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    changed = deepcopy(
        targeted
    )

    changed[
        "temporal_design"
    ][
        field
    ] = False

    with pytest.raises(
        ValueError,
        match="temporal contract",
    ):
        validate_targeted_validation_contract(
            targeted_config=
                changed,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


def test_contract_rejects_result_reporting_and_freeze_drift():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    changed = deepcopy(
        targeted
    )

    changed[
        "feature_ablation"
    ][
        "feature_selection_from_results"
    ] = True

    with pytest.raises(
        ValueError,
        match="Feature selection",
    ):
        validate_targeted_validation_contract(
            targeted_config=
                changed,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )

    changed = deepcopy(
        targeted
    )

    changed[
        "reporting"
    ][
        "pool_all_project_rows_as_primary_result"
    ] = True

    with pytest.raises(
        ValueError,
        match="must not pool",
    ):
        validate_targeted_validation_contract(
            targeted_config=
                changed,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )

    changed = deepcopy(
        targeted
    )

    changed[
        "freeze_policy"
    ][
        "result_driven_changes_allowed"
    ] = True

    with pytest.raises(
        ValueError,
        match="Result-driven",
    ):
        validate_targeted_validation_contract(
            targeted_config=
                changed,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


def test_contract_rejects_expected_grid_drift():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    changes = {
        "projects": 4,
        "feature_sets": 2,
        "models": 2,
        "calibration_variants": 2,
        "maximum_conditions": 134,
    }

    for (
        field,
        value,
    ) in changes.items():
        changed = deepcopy(
            targeted
        )

        changed[
            "expected_grid"
        ][
            field
        ] = value

        with pytest.raises(
            ValueError,
            match="grid",
        ):
            validate_targeted_validation_contract(
                targeted_config=
                    changed,
                rq1_config=
                    rq1,
                model_config=
                    models,
                calibration_config=
                    calibration,
                core_features_config=
                    core,
            )


def test_f1_predictions_ignore_f2_and_nonstructural_f3_values():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    changed_development = (
        development.copy()
    )

    changed_evaluation = (
        evaluation.copy()
    )

    for data in (
        changed_development,
        changed_evaluation,
    ):
        data[
            "changed_entity_count"
        ] = (
            data[
                "changed_entity_count"
            ]
            + 10000
        )

        data[
            "recent_failure_density_10"
        ] = (
            1.0
            - data[
                "recent_failure_density_10"
            ]
        )

        data[
            "execution_time_trend_10"
        ] = (
            -100.0
            * data[
                "execution_time_trend_10"
            ]
        )

    original, _ = (
        run_targeted_rq1_model(
            development=development,
            evaluation=evaluation,
            feature_set_name="F1",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    changed, _ = (
        run_targeted_rq1_model(
            development=
                changed_development,
            evaluation=
                changed_evaluation,
            feature_set_name="F1",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    pd.testing.assert_frame_equal(
        original[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        changed[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        check_exact=True,
    )


def test_f1_f2_predictions_ignore_nonstructural_f3_values():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    changed_development = (
        development.copy()
    )

    changed_evaluation = (
        evaluation.copy()
    )

    for data in (
        changed_development,
        changed_evaluation,
    ):
        data[
            "recent_failure_density_10"
        ] = (
            1.0
            - data[
                "recent_failure_density_10"
            ]
        )

        data[
            "execution_time_trend_10"
        ] = (
            -100.0
            * data[
                "execution_time_trend_10"
            ]
        )

    original, _ = (
        run_targeted_rq1_model(
            development=development,
            evaluation=evaluation,
            feature_set_name="F1_F2",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    changed, _ = (
        run_targeted_rq1_model(
            development=
                changed_development,
            evaluation=
                changed_evaluation,
            feature_set_name="F1_F2",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    pd.testing.assert_frame_equal(
        original[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        changed[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        check_exact=True,
    )


def test_temporal_partitions_are_identical_across_feature_sets():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    diagnostics = []

    for feature_set_name in (
        TARGETED_FEATURE_SET_ORDER
    ):
        _, current = (
            run_targeted_rq1_model(
                development=
                    development,
                evaluation=
                    evaluation,
                feature_set_name=
                    feature_set_name,
                model_name=
                    "logistic_regression",
                targeted_config=
                    targeted,
                rq1_config=
                    rq1,
                model_config=
                    models,
                calibration_config=
                    calibration,
                core_features_config=
                    core,
            )
        )

        diagnostics.append(
            current
        )

    temporal_fields = [
        "status",
        "model_fit_rows",
        "calibration_rows",
        "evaluation_rows",
        "model_fit_builds",
        "calibration_builds",
        "evaluation_builds",
        "model_fit_build_min",
        "model_fit_build_max",
        "calibration_build_min",
        "calibration_build_max",
        "evaluation_build_min",
        "evaluation_build_max",
        "initial_model_fit_builds",
        "expanded_by_builds",
        "calibration_positive_rows",
        "calibration_negative_rows",
        "calibration_failing_builds",
        "calibration_passing_builds",
    ]

    reference = {
        field:
            diagnostics[
                0
            ][
                field
            ]
        for field
        in temporal_fields
    }

    for current in (
        diagnostics[
            1:
        ]
    ):
        assert {
            field:
                current[
                    field
                ]
            for field
            in temporal_fields
        } == reference


def test_evaluation_labels_do_not_change_targeted_probabilities():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    changed_evaluation = (
        evaluation.copy()
    )

    changed_evaluation[
        "failed"
    ] = (
        1
        - changed_evaluation[
            "failed"
        ]
    )

    original, _ = (
        run_targeted_rq1_model(
            development=development,
            evaluation=evaluation,
            feature_set_name=
                "F1_F2_F3",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    changed, _ = (
        run_targeted_rq1_model(
            development=development,
            evaluation=
                changed_evaluation,
            feature_set_name=
                "F1_F2_F3",
            model_name=
                "logistic_regression",
            targeted_config=
                targeted,
            rq1_config=
                rq1,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
    )

    pd.testing.assert_frame_equal(
        original[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        changed[
            [
                "p_uncalibrated",
                "p_platt",
                "p_isotonic",
            ]
        ],
        check_exact=True,
    )


def test_insufficient_support_keeps_only_uncalibrated_probability():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    development = make_partition(
        builds=range(
            1,
            131,
        ),
        failure_builds={
            10,
            20,
        },
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
        },
    )

    (
        predictions,
        diagnostics,
    ) = run_targeted_rq1_model(
        development=development,
        evaluation=evaluation,
        feature_set_name="F1",
        model_name=
            "logistic_regression",
        targeted_config=
            targeted,
        rq1_config=
            rq1,
        model_config=
            models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    assert (
        diagnostics[
            "status"
        ]
        ==
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    )

    assert (
        diagnostics[
            "calibration_support_sufficient"
        ]
        is False
    )

    assert (
        diagnostics[
            "uncalibrated_available"
        ]
        is True
    )

    assert (
        diagnostics[
            "platt_available"
        ]
        is False
    )

    assert (
        diagnostics[
            "isotonic_available"
        ]
        is False
    )

    assert predictions[
        "p_uncalibrated"
    ].between(
        0.0,
        1.0,
    ).all()

    assert predictions[
        "p_platt"
    ].isna().all()

    assert predictions[
        "p_isotonic"
    ].isna().all()


def test_targeted_contract_reuses_frozen_rq1_contract():
    (
        targeted,
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    changed = deepcopy(
        rq1
    )

    changed[
        "preprocessing"
    ][
        "evaluation_period_may_fit_preprocessing"
    ] = True

    with pytest.raises(
        ValueError,
        match="Evaluation rows",
    ):
        validate_targeted_validation_contract(
            targeted_config=
                targeted,
            rq1_config=
                changed,
            model_config=
                models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
