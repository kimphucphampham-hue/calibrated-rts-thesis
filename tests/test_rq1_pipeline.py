from copy import deepcopy
from pathlib import Path

import pytest

from src.tcp_ci.rq1_pipeline import (
    PROBABILITY_COLUMNS,
    RQ1_CALIBRATION_VARIANTS,
    RQ1_MODEL_NAMES,
    build_rq1_model,
    load_yaml,
    prediction_columns,
    validate_rq1_integration_contract,
)


CONFIG_DIR = Path("configs")


EXPECTED_FEATURES = [
    "failure_rate_history",
    "recent_failure",
    "consecutive_failures",
    "consecutive_passes",
    "test_age_builds",
    "time_since_last_failure_builds",
    "execution_time_mean_history_ms",
    "changed_entity_count",
    "build_index",
    "recent_failure_density_10",
    "execution_time_trend_10",
]


def load_configs():
    rq1 = load_yaml(
        CONFIG_DIR
        / "rq1_v1.yaml"
    )

    models = load_yaml(
        CONFIG_DIR
        / "rq1_models_v1.yaml"
    )

    calibration = load_yaml(
        CONFIG_DIR
        / "calibration_v1.yaml"
    )

    core_features = load_yaml(
        CONFIG_DIR
        / "core_features_v1.yaml"
    )

    return (
        rq1,
        models,
        calibration,
        core_features,
    )


def test_rq1_integration_contract_matches_freeze():
    (
        rq1,
        models,
        calibration,
        core_features,
    ) = load_configs()

    features = (
        validate_rq1_integration_contract(
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core_features,
        )
    )

    assert features == EXPECTED_FEATURES

    assert list(
        RQ1_MODEL_NAMES
    ) == [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]

    assert list(
        RQ1_CALIBRATION_VARIANTS
    ) == [
        "uncalibrated",
        "platt",
        "isotonic",
    ]


def test_prediction_output_contract_is_frozen():
    assert PROBABILITY_COLUMNS == {
        "uncalibrated":
            "p_uncalibrated",

        "platt":
            "p_platt",

        "isotonic":
            "p_isotonic",
    }

    assert prediction_columns() == [
        "project_id",
        "build_id",
        "build_index",
        "test_id",
        "failed",
        "p_uncalibrated",
        "p_platt",
        "p_isotonic",
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ],
)
def test_build_rq1_model_supports_frozen_models(
    model_name,
):
    (
        rq1,
        models,
        calibration,
        core_features,
    ) = load_configs()

    features = (
        validate_rq1_integration_contract(
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core_features,
        )
    )

    pipeline = build_rq1_model(
        model_name=model_name,
        feature_columns=features,
        model_config=models,
    )

    assert list(
        pipeline.named_steps.keys()
    ) == [
        "preprocess",
        "model",
    ]


def test_build_rq1_model_rejects_unknown_model():
    (
        rq1,
        models,
        calibration,
        core_features,
    ) = load_configs()

    features = (
        validate_rq1_integration_contract(
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core_features,
        )
    )

    with pytest.raises(
        ValueError,
        match="Unsupported RQ1 model",
    ):
        build_rq1_model(
            model_name="svm",
            feature_columns=features,
            model_config=models,
        )


def test_contract_rejects_retraining():
    (
        rq1,
        models,
        calibration,
        core_features,
    ) = load_configs()

    changed = deepcopy(
        rq1
    )

    changed[
        "pilot_retraining"
    ][
        "enabled"
    ] = True

    with pytest.raises(
        ValueError,
        match="retraining",
    ):
        validate_rq1_integration_contract(
            rq1_config=changed,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core_features,
        )


def test_contract_rejects_calibration_variant_drift():
    (
        rq1,
        models,
        calibration,
        core_features,
    ) = load_configs()

    changed = deepcopy(
        calibration
    )

    changed["methods"] = [
        "uncalibrated",
        "platt",
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Calibration configuration "
            "does not match"
        ),
    ):
        validate_rq1_integration_contract(
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                changed,
            core_features_config=
                core_features,
        )


def test_contract_rejects_preprocessing_leakage():
    (
        rq1,
        models,
        calibration,
        core_features,
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
        validate_rq1_integration_contract(
            rq1_config=changed,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core_features,
        )
